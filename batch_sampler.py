import os
import numpy as np
import torch
from torch.utils.data import Dataset

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
