import os
import json
import torch
from transformers import AutoTokenizer

def load_tensor_file(tensor_path: str):
    encoded = torch.load(tensor_path)
    input_ids = encoded.get("input_ids")
    print(f"[INFO] Loaded tensor file: {tensor_path}")
    print(f"[INFO] input_ids shape: {input_ids.shape}")
    return input_ids

def decode_and_debug_tensor(input_ids, model_id: str, start_idx: int = 0, end_idx: int = 500):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    sliced_input_ids = input_ids[:, start_idx:end_idx]
    print("[Input IDs Slice]")
    print(sliced_input_ids[0].tolist())

    decoded_text = tokenizer.decode(sliced_input_ids[0], skip_special_tokens=True)
    print("[Decoded Text with skip_special_tokens=True]")
    print(decoded_text)

    decoded_text_full = tokenizer.decode(sliced_input_ids[0], skip_special_tokens=False)
    print("[Decoded Text with skip_special_tokens=False]")
    print(decoded_text_full)

if __name__ == "__main__":
    TENSOR_FILE = "encoded_context.pt"
    MODEL_ID = "davidkim205/ko-gemma-2-9b-it"

    input_ids = load_tensor_file(TENSOR_FILE)

    print("\n=== Debugging Decoding ===")
    decode_and_debug_tensor(input_ids, MODEL_ID, start_idx=0, end_idx=500)
