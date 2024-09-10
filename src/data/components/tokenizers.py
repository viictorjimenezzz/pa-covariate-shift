from typing import Any
from transformers import AutoTokenizer

def get_tokenizer(model: str):
    return AutoTokenizer.from_pretrained(model)

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors='pt')
def collate_fn_nlp(batch: Any):
    collated_0 = data_collator([
        el['0'] for el in batch
    ])
    collated_1 = data_collator([
        el['1'] for el in batch
    ])

    return {
        '0': (
            collated_0["input_ids"], collated_0['labels']
        ),
        '1': (
            collated_1["input_ids"], collated_1['labels']
        )
    }
