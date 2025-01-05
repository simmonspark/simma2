# train.py

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import argparse
from config import GemmaConfig
from layers.final_model import GemmaForCausalLM
from torch.cuda.amp import GradScaler, autocast  # AMP 관련 임포트
from torch.nn.utils import clip_grad_norm_

# ====================================================
# 커스텀 Dataset 정의
# ====================================================
class CausalLMDataset(Dataset):
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir (str): .npz 파일들이 저장된 디렉토리 경로
        """
        self.data_dir = data_dir
        self.file_list = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.npz')]
        self.file_list.sort()  # 파일 이름 순서대로 정렬 (선택 사항)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path, allow_pickle=True)
        input_ids = torch.tensor(data['input_ids'], dtype=torch.long)
        labels = torch.tensor(data['labels'], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': labels
        }

# ====================================================
# 학습 함수 정의
# ====================================================
def train(args):
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터셋 및 데이터로더 초기화
    dataset = CausalLMDataset(data_dir=args.data_dir)
    dataloader = dataset

    # 모델 초기화
    config = GemmaConfig()
    model = GemmaForCausalLM(config)
    model.to(device)

    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 스케줄러 (옵션: 학습률 스케줄링)
    if args.use_scheduler:
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    else:
        scheduler = None

    # 체크포인트 디렉토리 생성
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # AMP 스케일러 초기화
    scaler = GradScaler()

    # 학습 루프
    for epoch in range(1, 100):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.num_epochs}")

        # 스텝 카운터를 위한 enumerate 사용 (1부터 시작)
        for step, batch in enumerate(progress_bar, 1):
            input_ids = batch['input_ids'][0].unsqueeze(0).to(device)  # (batch_size=1, seq_len)
            labels = batch['labels'][0].unsqueeze(0).to(device)  # (batch_size=1, seq_len)

            # 마스크 생성 (float32로 변경, -1e4 사용)
            seq_len = input_ids.size(1)
            mask = torch.triu(
                torch.ones((seq_len, seq_len), device=device, dtype=torch.float32),
                diagonal=1
            ) * -1e4  # (seq_len, seq_len)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            mask = mask.expand(input_ids.size(0), -1, -1, -1)  # (batch_size=1, 1, seq_len, seq_len)

            optimizer.zero_grad()

            with autocast():  # 자동 혼합 정밀도 컨텍스트 매니저
                logits = model(input_ids, mask)  # (batch_size, seq_len, vocab_size)
                loss = model.compute_loss(logits, labels, ignore_index=0)

            scaler.scale(loss).backward()  # 손실 스케일링 및 역전파

            # Gradient Clipping 적용 (스케일링 된 기울기에 적용)
            scaler.unscale_(optimizer)  # 스케일러 언스케일
            clip_grad_norm_(model.parameters(), max_norm=0.8)  # 기울기 클리핑

            scaler.step(optimizer)  # 옵티마이저 스텝
            scaler.update()  # 스케일러 업데이트

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

            # 100 스텝 후 저장 및 에포크 종료
            if step >= 100:
                checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch}_step_{step}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"\nSaved checkpoint: {checkpoint_path} at epoch {epoch}, step {step}")
                break  # 다음 에포크로 넘어감

        avg_epoch_loss = epoch_loss / min(len(dataloader), 100)  # 실제 처리한 스텝 수로 나눔
        print(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.4f}")

        # 스케줄러 스텝
        if scheduler:
            scheduler.step()

    print("Training completed.")

# ====================================================
# 검증 함수 정의 (선택 사항)
# ====================================================
def validate(args, model, dataloader, device):
    model.eval()
    total_loss = 0.0
    scaler = GradScaler()  # 검증에서도 스케일러 사용 (선택 사항)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            mask = torch.triu(
                torch.ones(
                    (input_ids.size(0), 1, input_ids.size(1), input_ids.size(1)),
                    device=device,
                    dtype=torch.float16  # 마스크를 float16으로 유지
                ),
                diagonal=1
            ) * float('-inf')

            with autocast():
                logits = model(input_ids, mask)
                # logits = logits.half()  # AMP 사용 시 필요 없음
                loss = model.compute_loss(logits, labels, ignore_index=0)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Average Loss: {avg_loss:.4f}")
    return avg_loss

# ====================================================
# 메인 함수 및 인자 파서 정의
# ====================================================
def main():
    parser = argparse.ArgumentParser(description="Train GemmaForCausalLM Model")
    parser.add_argument('--data_dir', type=str, default='/media/sien/media/data/train_data/', help='디렉토리 내 .npz 데이터 파일 경로')
   # parser.add_argument('--batch_size', type=int, default=2, help='배치 크기')
    parser.add_argument('--num_epochs', type=int, default=10, help='학습 에포크 수')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='학습률')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='가중치 감쇠')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader의 num_workers')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/', help='모델 체크포인트 저장 디렉토리')
    parser.add_argument('--use_scheduler', action='store_true', help='학습률 스케줄러 사용 여부')
    parser.add_argument('--scheduler_step_size', type=int, default=1, help='스케줄러 스텝 사이즈')
    parser.add_argument('--scheduler_gamma', type=float, default=0.95, help='스케줄러 감마 값')

    args = parser.parse_args()

    # 학습 시작
    train(args)

if __name__ == "__main__":
    main()
