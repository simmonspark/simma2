# config.py

class GemmaConfig:
    def __init__(self):
        self.vocab_size = 256000
        self.max_position_embeddings = 4096  # 이전 1024에서 4096으로 증가
        self.num_hidden_layers = 18
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
