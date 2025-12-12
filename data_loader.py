"""
Data preparation module for paraphrase generation
Downloads and prepares datasets for training
"""
import os
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from config import DATA_DIR, MODEL_CONFIG
import pandas as pd
from tqdm import tqdm


class ParaphraseDataLoader:
    """Loads and prepares paraphrase datasets"""
    
    def __init__(self, model_name=None):
        self.model_name = model_name or MODEL_CONFIG["base_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    def load_datasets(self):
        """Load multiple paraphrase datasets"""
        print("Loading paraphrase datasets...")
        
        datasets = []
        
        # 1. PAWS (Paraphrase Adversaries from Word Scrambling)
        try:
            paws = load_dataset("paws", "labeled_final", split="train")
            paws_filtered = paws.filter(lambda x: x["label"] == 1)  # Only paraphrases
            paws_formatted = paws_filtered.map(
                lambda x: {
                    "input_text": x["sentence1"],
                    "target_text": x["sentence2"]
                },
                remove_columns=paws_filtered.column_names  # Remove all original columns
            )
            datasets.append(paws_formatted)
            print(f"✓ Loaded PAWS dataset: {len(paws_formatted)} examples")
        except Exception as e:
            print(f"✗ Could not load PAWS: {e}")
        
        # 2. Quora Question Pairs
        try:
            quora = load_dataset("quora", split="train[:15000]")  # Reduced for faster loading
            quora_filtered = quora.filter(lambda x: x["is_duplicate"] == True)
            quora_formatted = quora_filtered.map(
                lambda x: {
                    "input_text": x["questions"]["text"][0],
                    "target_text": x["questions"]["text"][1]
                },
                remove_columns=quora_filtered.column_names  # Remove all original columns
            )
            datasets.append(quora_formatted)
            print(f"✓ Loaded Quora dataset: {len(quora_formatted)} examples")
        except Exception as e:
            print(f"✗ Could not load Quora: {e}")
        
        # 3. MRPC (Microsoft Research Paraphrase Corpus)
        try:
            mrpc = load_dataset("glue", "mrpc", split="train")
            mrpc_filtered = mrpc.filter(lambda x: x["label"] == 1)
            mrpc_formatted = mrpc_filtered.map(
                lambda x: {
                    "input_text": x["sentence1"],
                    "target_text": x["sentence2"]
                },
                remove_columns=mrpc_filtered.column_names  # Remove all original columns
            )
            datasets.append(mrpc_formatted)
            print(f"✓ Loaded MRPC dataset: {len(mrpc_formatted)} examples")
        except Exception as e:
            print(f"✗ Could not load MRPC: {e}")
        # 4. ParaNMT (Synthetic paraphrase dataset) - subset
        try:
            paranmt = load_dataset("embedding-data/sentence-compression", split="train[:8000]")  # Reduced
            paranmt_formatted = paranmt.map(
                lambda x: {
                    "input_text": x["set"][0] if len(x["set"]) > 0 else "",
                    "target_text": x["set"][1] if len(x["set"]) > 1 else ""
                },
                remove_columns=paranmt.column_names  # Remove all original columns
            )
            # Filter empty examples
            paranmt_formatted = paranmt_formatted.filter(
                lambda x: len(x["input_text"]) > 20 and len(x["target_text"]) > 20
            )
            datasets.append(paranmt_formatted)
            print(f"✓ Loaded ParaNMT dataset: {len(paranmt_formatted)} examples")
        except Exception as e:
            print(f"✗ Could not load ParaNMT: {e}")
            print(f"✗ Could not load ParaNMT: {e}")
        
        if not datasets:
            raise ValueError("No datasets could be loaded!")
        
        # Concatenate all datasets
        combined_dataset = concatenate_datasets(datasets)
        print(f"\n✓ Total training examples: {len(combined_dataset)}")
        
        return combined_dataset
    
    def preprocess_function(self, examples):
        """Preprocess data for T5/BART model with length filtering"""
        from config import INPUT_CONSTRAINTS
        
        # Filter examples by length ratio (65-85% range for training)
        filtered_inputs = []
        filtered_targets = []
        
        for inp, tgt in zip(examples["input_text"], examples["target_text"]):
            inp_words = len(inp.split())
            tgt_words = len(tgt.split())
            
            if inp_words > 0:
                ratio = tgt_words / inp_words
                # Keep examples in 65-85% range (slightly wider for training diversity)
                if 0.65 <= ratio <= 0.85:
                    filtered_inputs.append(inp)
                    filtered_targets.append(tgt)
        
        # If all filtered out, keep originals (shouldn't happen often)
        if not filtered_inputs:
            filtered_inputs = examples["input_text"]
            filtered_targets = examples["target_text"]
        
        # T5 requires task prefix
        if "t5" in self.model_name.lower():
            inputs = ["paraphrase: " + text for text in filtered_inputs]
        else:
            inputs = filtered_inputs
        
        targets = filtered_targets
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=MODEL_CONFIG["max_input_length"],
            truncation=True,
            padding="max_length"
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            targets,
            max_length=MODEL_CONFIG["max_output_length"],
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def prepare_datasets(self):
        """Load and prepare train/validation datasets"""
        from config import TRAINING_CONFIG
        
        # Load combined dataset
        dataset = self.load_datasets()
        
        # Shuffle and select columns
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select_columns(["input_text", "target_text"])
        
        # Limit dataset size if specified (for faster training)
        max_samples = TRAINING_CONFIG.get("max_train_samples")
        if max_samples and len(dataset) > max_samples:
            print(f"⚡ Limiting dataset to {max_samples} samples for faster training")
            dataset = dataset.select(range(max_samples))
        
        # Split into train/validation
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
        
        print("\nTokenizing datasets...")
        # Tokenize
        tokenized_train = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train data"
        )
        
        tokenized_val = val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation data"
        )
        
        print(f"✓ Prepared {len(tokenized_train)} training examples")
        print(f"✓ Prepared {len(tokenized_val)} validation examples")
        
        return tokenized_train, tokenized_val, train_dataset, val_dataset


def create_synthetic_long_paraphrases():
    """Create synthetic long-form paraphrase examples for fine-tuning"""
    # This function can be used to create longer paraphrase examples
    # by combining multiple sentences or using data augmentation
    
    examples = [
        {
            "input_text": "Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data. These systems improve their performance over time without being explicitly programmed. The algorithms identify patterns in data and make decisions based on those patterns.",
            "target_text": "A branch of AI called machine learning concentrates on developing systems capable of learning from information. Over time, these systems enhance their capabilities without requiring explicit programming. The computational methods detect data patterns and base their decisions on these identified patterns."
        },
        # Add more examples...
    ]
    
    return Dataset.from_pandas(pd.DataFrame(examples))


if __name__ == "__main__":
    # Test data loading
    loader = ParaphraseDataLoader()
    train_data, val_data, raw_train, raw_val = loader.prepare_datasets()
    
    print("\n" + "="*50)
    print("Data preparation complete!")
    print("="*50)
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Show example
    print("\nExample from training data:")
    print(f"Input: {raw_train[0]['input_text']}")
    print(f"Target: {raw_train[0]['target_text']}")
