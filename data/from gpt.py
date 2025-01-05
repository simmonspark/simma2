import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple, List, Union
import torch.utils.checkpoint as checkpoint
import os
from torch.cuda.amp import autocast, GradScaler



##################################################
# GemmaConfig (조정된 설정)
##################################################
class GemmaConfig:
    def __init__(self):
        self.vocab_size = 256000
        self.max_position_embeddings = 4096  # 이전 1024에서 4096으로 증가
        self.num_hidden_layers = 16
        self.num_attention_heads = 4  # 유지
        self.num_key_value_heads = 1
        self.hidden_size = 1024  # 768에서 1024로 변경 (4 * 256)
        self.intermediate_size = 4096  # 일반적으로 4 * hidden_size
        self.head_dim = 256
        self.rms_norm_eps = 1e-6
        self.dtype = "float16"  # 유지
        self.quant = False
        self.tokenizer = "tokenizer/tokenizer.model"
        self.attn_types = None
        self.sliding_window_size = None
        self.final_logit_softcapping = None
        self.attn_logit_softcapping = None
        self.query_pre_attn_scalar = None
        self.use_pre_ffw_norm = False
        self.use_post_ffw_norm = False

##################################################
# Embedding 클래스
##################################################
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), dtype=torch.float32))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


##################################################
# Rotary Positional Encoding (RoPE)
##################################################
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device: torch.device = torch.device('cpu')):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end).float()
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(device)  # (end, dim/2) 복소수
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_real: torch.Tensor, freqs_imag: torch.Tensor) -> torch.Tensor:
    # x: (batch_size, seq_len, num_heads, head_dim)
    # freqs_real: (seq_len, head_dim/2)
    # freqs_imag: (seq_len, head_dim/2)
    bsz, seqlen, nheads, hdim = x.shape
    x_ = x.float().reshape(bsz, seqlen, nheads, hdim // 2, 2)  # (bsz, seqlen, nheads, head_dim/2, 2)

    # 실수와 허수 부분 분리
    real = x_[..., 0]  # (bsz, seqlen, nheads, head_dim/2)
    imag = x_[..., 1]  # (bsz, seqlen, nheads, head_dim/2)

    # 브로드캐스팅을 위해 차원 추가
    freqs_real = freqs_real.unsqueeze(0).unsqueeze(2)  # (1, seqlen, 1, head_dim/2)
    freqs_imag = freqs_imag.unsqueeze(0).unsqueeze(2)  # (1, seqlen, 1, head_dim/2)

    # RoPE 적용
    rotated_real = real * freqs_real - imag * freqs_imag
    rotated_imag = real * freqs_imag + imag * freqs_real

    # 다시 합침
    rotated = torch.stack((rotated_real, rotated_imag), dim=-1).reshape(bsz, seqlen, nheads, hdim)

    return rotated


##################################################
# RMSNorm 클래스
##################################################
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))  # float32으로 설정

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight


##################################################
# Self-Attention 클래스
##################################################
class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        assert self.hidden_size == self.num_heads * self.head_dim, \
            f"hidden_size ({self.hidden_size}) must be equal to num_heads ({self.num_heads}) * head_dim ({self.head_dim})"

        # QKV를 한 번에 계산
        self.qkv_proj = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.rms_norm_eps = config.rms_norm_eps
        self.scaling = (self.head_dim) ** -0.5

    def forward(self, hidden_states: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor):
        # hidden_states: (batch_size, seq_len, hidden_size)
        # freqs_cis: (freqs_real, freqs_imag)
        freqs_real, freqs_imag = freqs_cis

        bsz, seqlen, _ = hidden_states.shape

        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(bsz, seqlen, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # 각 (bsz, seqlen, num_heads, head_dim)

        # RoPE 적용
        q = apply_rotary_emb(q, freqs_real, freqs_imag)
        k = apply_rotary_emb(k, freqs_real, freqs_imag)

        # q, k shape: (bsz, seqlen, nheads, head_dim)
        q = q.transpose(1, 2)  # (bsz, nheads, seqlen, head_dim)
        k = k.transpose(1, 2)  # (bsz, nheads, seqlen, head_dim)
        v = v.transpose(1, 2)  # (bsz, nheads, seqlen, head_dim)

        # Scaled Dot-Product
        q = q * self.scaling
        scores = torch.matmul(q, k.transpose(-1, -2))  # (bsz, nheads, seqlen, seqlen)
        scores = scores + mask
        attn_probs = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)  # (bsz, nheads, seqlen, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (bsz, seqlen, nheads, head_dim)
        attn_output = attn_output.reshape(bsz, seqlen, self.hidden_size)

        # Output projection
        output = self.o_proj(attn_output)
        return output


##################################################
# MLP 클래스
##################################################
class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (bsz, seq_len, hidden_size)
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fused = gate * up
        out = self.down_proj(fused)
        return out


##################################################
# Decoder Layer (Self-Attn + MLP)
##################################################
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.self_attn = GemmaAttention(config)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor):
        # Self-Attn
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        def custom_forward_attn(*inputs):
            return self.self_attn(*inputs)

        attn_out = checkpoint.checkpoint(custom_forward_attn, hidden_states, freqs_cis, mask)
        hidden_states = residual + attn_out

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        def custom_forward_mlp(*inputs):
            return self.mlp(*inputs)

        mlp_out = checkpoint.checkpoint(custom_forward_mlp, hidden_states)
        hidden_states = residual + mlp_out

        return hidden_states


##################################################
# Main Decoder (Stack of Layers)
##################################################
class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,  # (bsz, seq_len, hidden_size)
            freqs_cis: Tuple[torch.Tensor, torch.Tensor],  # (freqs_real, freqs_imag)
            mask: torch.Tensor,  # (bsz, 1, seq_len, seq_len)
    ):
        for layer in self.layers:
            hidden_states = layer(hidden_states, freqs_cis, mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states


##################################################
# Final Causal LM
##################################################
class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        # 임베딩
        self.embedder = Embedding(config.vocab_size, config.hidden_size)
        # 본체
        self.model = GemmaModel(config)
        # 로터리 임베딩 사전 계산
        rope_theta = 10000.0
        self.freqs_cis = precompute_freqs_cis(config.head_dim, config.max_position_embeddings * 2, theta=rope_theta,
                                              device='cuda')
        # freqs_real과 freqs_imag으로 분리
        self.freqs_real = self.freqs_cis.real.to(torch.float16)
        self.freqs_imag = self.freqs_cis.imag.to(torch.float16)
        # 최종 Linear를 두지 않고, embedder.weight를 matmul하기 위해 사용
        self.embed_weight = self.embedder.weight

    def forward(
            self,
            input_ids: torch.Tensor,  # (bsz, seq_len)
            mask: torch.Tensor  # (bsz, 1, seq_len, seq_len), causal mask
    ):
        """
        간단히, 전체 시퀀스를 한 번에 forward.
        - KV 캐시 없이, 모든 토큰 함께 처리
        - logits = hidden @ embed_weight^T
        """
        bsz, seq_len = input_ids.shape
        # RoPE 위치 인덱스
        # 만약 위치별로 [0..seq_len-1]이라고 가정
        positions = torch.arange(seq_len, device=input_ids.device)
        # freqs_cis 중 필요한 부분만 추출
        freqs_real_slice = self.freqs_real.index_select(0, positions)
        freqs_imag_slice = self.freqs_imag.index_select(0, positions)

        # 1) 임베딩
        hidden_states = self.embedder(input_ids)  # (bsz, seq_len, hidden_size)
        # 2) sqrt(hidden_size)로 스케일링
        normalizer = (float(self.config.hidden_size) ** 0.5)
        hidden_states = hidden_states * normalizer
        # 3) 본체
        hidden_states = self.model(hidden_states, (freqs_real_slice, freqs_imag_slice),
                                   mask)  # (bsz, seq_len, hidden_size)
        # 4) logits
        #    (bsz, seq_len, hidden_size) x (hidden_size, vocab_size) -> (bsz, seq_len, vocab_size)
        logits = torch.matmul(hidden_states, self.embed_weight.t())
        return logits

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = 0):
        """
        손실 계산 함수.

        Args:
            logits (torch.Tensor): 모델의 출력 로짓, shape은 (batch_size, seq_len, vocab_size)
            labels (torch.Tensor): 정답 레이블, shape은 (batch_size, seq_len)
            ignore_index (int, optional): 무시할 인덱스 값. 기본값은 0.

        Returns:
            torch.Tensor: 계산된 손실 값
        """
        # logits 시퀀스를 한 시퀀스 앞으로 시프트
        shift_logits = logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # (batch_size * (seq_len-1), vocab_size)

        # labels 시퀀스를 한 시퀀스 앞으로 시프트하여 targets 생성
        shift_labels = labels[:, 1:].contiguous().view(-1)  # (batch_size * (seq_len-1))

        # Cross Entropy Loss 계산
        loss = F.cross_entropy(
            shift_logits,  # (N, C)
            shift_labels,  # (N,)
            ignore_index=ignore_index
        )
        return loss


##################################################
# 모델 파라미터 수 확인 함수
##################################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


##################################################
# 데이터 로딩 및 텐서 크기 확인 함수
##################################################
def example_forward_with_zero_tensors(config: GemmaConfig, model: GemmaForCausalLM, batch_size: int, seq_len: int):
    # 데이터 로딩 및 텐서 크기 확인
    # 여기서는 예제로 랜덤 텐서를 사용하지만, 실제 데이터로 교체해야 합니다.
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), device='cuda', dtype=torch.long)
    labels = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), device='cuda', dtype=torch.long)

    print(f"input_ids shape: {input_ids.shape}")
    print(f"labels shape: {labels.shape}")

    # 마스크 생성
    mask = torch.zeros((batch_size, 1, seq_len, seq_len), device='cuda', dtype=torch.float16)
    mask = torch.triu(mask, diagonal=1) * float('-inf')  # 상삼각형 마스크

    # Mixed Precision 및 Gradient Accumulation 준비
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    accumulation_steps = 16  # 예: 배치 누적 단계 수

    model.train()

    for step in range(accumulation_steps):
        optimizer.zero_grad()

        with autocast():
            logits = model.forward(input_ids, mask)
            loss = model.compute_loss(logits, labels)
            loss = loss / accumulation_steps  # 손실 나누기

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            print(f"Step {step + 1}/{accumulation_steps}, Loss: {loss.item() * accumulation_steps:.4f}")

    # 메모리 사용량 확인
    print("Logit Shape:", logits.shape)
    print(torch.cuda.memory_summary(device=None, abbreviated=False))


##################################################
# 메인 함수
##################################################
import torch
from torch.cuda.amp import autocast, GradScaler

# 모델 및 Config 정의는 이전 코드 참조

if __name__ == "__main__":
    # GPU 메모리 정리
    torch.cuda.empty_cache()

    # Config 및 모델 초기화
    config = GemmaConfig()
    model = GemmaForCausalLM(config).to("cuda")

    # 모델 파라미터 수 출력
    print(f"Total trainable parameters: {count_parameters(model):,}")

    # 옵티마이저 및 스케일러 초기화
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    # 배치 크기 및 시퀀스 길이 설정
    batch_size = 2  # 실제 훈련 시 조정
    seq_len = 2048  # 실제 훈련 시 조정

    # 입력 데이터 생성 (랜덤 데이터)
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), device='cuda', dtype=torch.long)
    labels = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), device='cuda', dtype=torch.long)

    # 마스크 생성
    mask = torch.zeros((batch_size, 1, seq_len, seq_len), device='cuda', dtype=torch.float16)
    mask = torch.triu(mask, diagonal=1) * float('-inf')  # 상삼각형 마스크

    # 훈련 루프
    model.train()
    for step in range(10):  # 1 스텝만 실행
        optimizer.zero_grad()

        with autocast():
            logits = model(input_ids, mask)
            loss = model.compute_loss(logits, labels)

        scaler.scale(loss).backward()  # Backward
        scaler.step(optimizer)  # Optimizer step
        scaler.update()  # Update scaler

        print(f"Step: {step + 1}, Loss: {loss.item():.4f}")

    # GPU 메모리 사용량 출력
    allocated_memory = torch.cuda.memory_allocated() / 1024 ** 3  # Convert to GB
    reserved_memory = torch.cuda.memory_reserved() / 1024 ** 3  # Convert to GB

    print(f"GPU Memory Allocated: {allocated_memory:.2f} GB")
    print(f"GPU Memory Reserved: {reserved_memory:.2f} GB")

    # 상세 메모리 상태 출력
    print(torch.cuda.memory_summary(device=None, abbreviated=False))