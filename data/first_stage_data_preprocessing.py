import os
import json
import torch
import concurrent.futures
from transformers import AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

def extract_contents_from_file(json_path: str):
    results = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data_info_list = data.get("data_info", [])
        for item in data_info_list:
            contents = item.get("contents", "")
            if not contents.strip():
                continue
            results.append(f"<bos>{contents}<eos>")
    except Exception as e:
        print(f"[ERROR] 파일 읽기 실패: {json_path}, 에러: {e}")
    return results

def chunk_text_list(text_list, chunk_size):
    chunks = []
    current_chunk = []
    for i, txt in enumerate(text_list, start=1):
        current_chunk.append(txt)
        if (i % chunk_size == 0) or (i == len(text_list)):
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    return chunks

def tokenize_chunk(chunk_text, model_id):
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoded = tokenizer(chunk_text, return_tensors="pt")
    return encoded["input_ids"], encoded["attention_mask"]

def get_context_and_tokenize_in_chunks_parallel(
    base_dir: str,
    model_id: str,
    save_path: str,
    chunk_size: int = 1000,
    num_workers: int = 4
):
    json_paths = []
    for root, dirs, files in os.walk(base_dir):
        for name in files:
            if name.lower().endswith(".json"):
                json_paths.append(os.path.join(root, name))

    all_texts = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(extract_contents_from_file, path) for path in json_paths]
        with tqdm(total=len(futures), desc="Reading JSON files") as pbar:
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                all_texts.extend(res)
                pbar.update(1)

    if not all_texts:
        print("[DEBUG] No valid contents found in JSON files.")
        return

    print(f"[DEBUG] Total valid contents: {len(all_texts)}")
    chunks = chunk_text_list(all_texts, chunk_size)
    print(f"[DEBUG] Total chunks created: {len(chunks)}")
    print(f"[DEBUG] First 5 Chunks: {chunks[:5]}")

    input_ids_list = []
    attention_masks_list = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results_iter = executor.map(tokenize_chunk, chunks, [model_id]*len(chunks))
        with tqdm(total=len(chunks), desc="Tokenizing Chunks") as pbar:
            for input_ids, attention_mask in results_iter:
                input_ids_list.append(input_ids)
                attention_masks_list.append(attention_mask)
                pbar.update(1)

    if not input_ids_list:
        print("[DEBUG] No valid input_ids generated during tokenization.")
        return

    input_ids_concat = torch.cat(input_ids_list, dim=1)
    attention_mask_concat = torch.cat(attention_masks_list, dim=1)
    encoded_full = {
        "input_ids": input_ids_concat,
        "attention_mask": attention_mask_concat
    }
    torch.save(encoded_full, save_path)
    print(f"[INFO] 최종 토큰화된 텐서를 '{save_path}'에 저장 완료.")
    return encoded_full

if __name__ == "__main__":
    BASE_DIR = "/media/sien/media/data/121.한국어 성능이 개선된 초거대AI 언어모델 개발 및 데이터/3.개방데이터/1.데이터/Training"
    MODEL_ID = "davidkim205/ko-gemma-2-9b-it"
    SAVE_PATH = "encoded_context.pt"

    encoded = get_context_and_tokenize_in_chunks_parallel(
        base_dir=BASE_DIR,
        model_id=MODEL_ID,
        save_path=SAVE_PATH,
        chunk_size=100,
        num_workers=10
    )
    if encoded:
        print("[INFO] 최종 input_ids shape:", encoded["input_ids"].shape)
