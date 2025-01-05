from transformers import AutoTokenizer

def test_specific_text(model_id: str, text: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoded = tokenizer(text, return_tensors="pt")
    decoded_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)

    print("[Original Text]")
    print(text)
    print("\n[Encoded Tokens]")
    print(encoded["input_ids"][0].tolist())
    print("\n[Decoded Text]")
    print(decoded_text)

if __name__ == "__main__":
    MODEL_ID = "davidkim205/ko-gemma-2-9b-it"
    test_text = '<bos>안녕 시발련들아! simma 이구요, 시언이의 말투를 그대로 Distill 해서 응답을 하기 위해 한국어를 공부하고. 있어요 앞으로 카카오톡 응대는 이 친구에게 시킬거에요<eos>'
    test_specific_text(MODEL_ID, test_text)
