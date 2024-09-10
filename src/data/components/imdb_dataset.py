from typing import Optional
from torch.utils.data import Dataset

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator

from src.data.components import MultienvDataset
from src.data.components.tokenizers import get_tokenizer

import random
from functools import partial

class IMDB_PADataset(MultienvDataset):
    def __init__(
            self,
            perturbation: Optional[str] = "levenshtein",
            intensity: Optional[int] = 1,
            seed: Optional[int] = 0
        ):
        assert perturbation in ["levenshtein", "removal", "adversarial", "adversarial-inverted"], "Perturbation method must be either 'levenshtein', 'removal', 'adversarial' or 'adversarial-inverted'."
        assert intensity >= 0, "Perturbation intensity must be non-negative."
        
        imdb = load_dataset("imdb")
        self.n_test = 25000
        self.data_collator = DefaultDataCollator(return_tensors='pt') 
        self.tokenizer = get_tokenizer(model="distilbert-base-uncased")

        # Clean dataset
        ds = imdb["test"].shuffle(seed=seed).select(
            [i for i in list(range(self.n_test))]
        )
        ds_original = ds.map(self._tokenize, batched=True)

        # Perturbed dataset
        self._tokenize_perturbed = partial(self._tokenize, perturbation=perturbation, intensity=intensity)
        ds_perturbed = ds.map(self._tokenize_perturbed, batched=True)

        super().__init__(
            dset_list = [ds_original, ds_perturbed]
        )

    @staticmethod
    def levenshtein_attack(text: str, num_changes: int):
        """Applies a Levenshtein-based attack on the text."""
        for _ in range(num_changes):
            if len(text) <= 2:
                break
            
            valid_indices = [i for i, char in enumerate(text) if char != '\\']
            if not valid_indices:
                break
            
            change_type = random.choice(['insert', 'delete', 'substitute'])
            pos = random.choice(valid_indices)
            if change_type == 'insert':
                char = random.choice('abcdefghijklmnopqrstuvwxyz')
                text = text[:pos] + char + text[pos:]
            elif change_type == 'delete' and len(text) > 1:
                text = text[:pos] + text[pos + 1:]
            elif change_type == 'substitute':
                char = random.choice('abcdefghijklmnopqrstuvwxyz')
                text = text[:pos] + char + text[pos + 1:]
        
        return text
    
    @staticmethod
    def removal_attack(text: str, num_words_to_remove: int):
        """Applies a word removal attack on the text."""
        words = text.split()
        if len(words) <= num_words_to_remove - 1:
            return random.choice(words)
        
        indices_to_remove = random.sample(range(len(words)), num_words_to_remove)
        return " ".join([
            word 
            for i, word in enumerate(words) if i not in indices_to_remove
        ])
    
    @staticmethod
    def adversarial_attack(text: str, label: int, num_changes: int, inverted: Optional[bool] = False):
        """Applies an adversarial-inspired attack on the text."""
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "horrible", "poor", "disappointing"]

        positive_label = 1 if inverted == False else 0
        words = text.split()
    
        if len(words) <= 1:
            return text
        
        valid_words_indices = [i for i, word in enumerate(words) if '\\' not in word]
        
        if not valid_words_indices:
            return text
        
        for _ in range(num_changes):
            if not valid_words_indices:
                break
            
            # Select a random word to replace
            word_to_replace_index = random.choice(valid_words_indices)
            
            # Replace based on the label
            if label == positive_label:  # Positive label, replace with a negative word
                replacement_word = random.choice(negative_words)
            else:  # Negative label, replace with a positive word
                replacement_word = random.choice(positive_words)
            
            # Perform the replacement
            words[word_to_replace_index] = replacement_word
            
            # Remove the chosen index from the valid indices to avoid repeated changes at the same position
            valid_words_indices.remove(word_to_replace_index)
        
        return " ".join(words)


    def _tokenize(self, item, perturbation: Optional[str] = None, intensity: Optional[int] = None):
        if perturbation == "levenshtein":
            perturbed_text = [
                self.levenshtein_attack(text, num_changes = int(intensity))
                for text in item['text']
            ]
        elif perturbation == "removal":
            perturbed_text = [
                self.removal_attack(text, num_words_to_remove = int(intensity))
                for text in item['text']
            ]
        elif perturbation == "adversarial":
            perturbed_text = [
                self.adversarial_attack(text, label = label, num_changes = int(intensity), inverted=False)
                for text, label in zip(item['text'], item['label'])
            ]
        elif perturbation == "adversarial-inverted":
            perturbed_text = [
                self.adversarial_attack(text, label = label, num_changes = int(intensity), inverted=True)
                for text, label in zip(item['text'], item['label'])
            ]

        else: # no perturbation
            perturbed_text = item["text"]
            
        return self.tokenizer(
            perturbed_text,
            truncation=True,
            padding="max_length",
            max_length=512
        )
    