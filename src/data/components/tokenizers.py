from transformers import AutoTokenizer

def get_tokenizer(model: str):
    return AutoTokenizer.from_pretrained(model)