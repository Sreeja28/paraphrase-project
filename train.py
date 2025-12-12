"""
Training script for Custom Paraphrase Generator (CPG)
Fine-tunes T5 or BART model on paraphrase datasets
"""
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from data_loader import ParaphraseDataLoader
from config import MODEL_CONFIG, TRAINING_CONFIG, MODELS_DIR
import numpy as np
from evaluate import load


class CPGTrainer:
    """Custom Paraphrase Generator Trainer"""
    
    def __init__(self, model_name=None, output_dir=None):
        self.model_name = model_name or MODEL_CONFIG["base_model"]
        self.output_dir = output_dir or str(MODELS_DIR / "cpg_model")
        
        print(f"Initializing CPG Trainer with {self.model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Load metrics
        self.bleu_metric = load("sacrebleu")
        self.rouge_metric = load("rouge")
        
        print(f"Model loaded: {self.model_name}")
        print(f"Parameters: {self.model.num_parameters():,}")
        
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Replace -100 in labels (used for padding)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        # Clip predictions to valid token ID range to avoid overflow
        predictions = np.clip(predictions, 0, self.tokenizer.vocab_size - 1)
        labels = np.clip(labels, 0, self.tokenizer.vocab_size - 1)
        
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean up whitespace
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        
        # Compute BLEU
        bleu_result = self.bleu_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )
        
        # Compute ROUGE
        rouge_result = self.rouge_metric.compute(
            predictions=decoded_preds,
            references=[label[0] for label in decoded_labels]
        )
        
        return {
            "bleu": bleu_result["score"],
            "rouge1": rouge_result["rouge1"],
            "rouge2": rouge_result["rouge2"],
            "rougeL": rouge_result["rougeL"],
        }
    
    def train(self, train_dataset, val_dataset):
        """Train the model"""
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Use single GPU only (CUDA_VISIBLE_DEVICES controls which GPU)
        print(f"Using single GPU training")
        
        # Training arguments for single GPU
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="steps",
            eval_steps=TRAINING_CONFIG["eval_steps"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
            per_device_eval_batch_size=TRAINING_CONFIG["batch_size"],
            num_train_epochs=TRAINING_CONFIG["num_epochs"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
            warmup_steps=TRAINING_CONFIG["warmup_steps"],
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
            save_steps=TRAINING_CONFIG["save_steps"],
            save_total_limit=3,
            logging_steps=TRAINING_CONFIG["logging_steps"],
            predict_with_generate=True,
            generation_max_length=MODEL_CONFIG["max_output_length"],
            generation_num_beams=4,
            fp16=TRAINING_CONFIG["fp16"] and torch.cuda.is_available(),
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            report_to="none",  # Disable tensorboard
            push_to_hub=False,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
        )
        
        # Set generation config for length control during evaluation
        # Ensure num_beams > 1 when using length_penalty to satisfy generation config validation
        self.model.config.num_beams = 4
        self.model.config.length_penalty = 0.8
        self.model.config.no_repeat_ngram_size = 2
        # Mirror to generation_config to avoid save_pretrained validation issues
        self.model.generation_config.num_beams = 4
        self.model.generation_config.length_penalty = 0.8
        self.model.generation_config.no_repeat_ngram_size = 2
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        print("\n" + "="*70)
        print("Starting training...")
        print("="*70)
        
        # Train
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print("\n" + "="*70)
        print("Training complete!")
        print("="*70)
        print(f"Model saved to: {self.output_dir}")
        
        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
        return trainer, metrics, eval_metrics


def main():
    """Main training function"""
    print("="*70)
    print("Custom Paraphrase Generator (CPG) Training Pipeline")
    print("="*70)
    
    # Step 1: Load and prepare data
    print("\n[Step 1/3] Loading and preparing datasets...")
    data_loader = ParaphraseDataLoader()
    train_dataset, val_dataset, _, _ = data_loader.prepare_datasets()
    
    # Step 2: Initialize trainer
    print("\n[Step 2/3] Initializing trainer...")
    trainer_obj = CPGTrainer()
    
    # Step 3: Train model
    print("\n[Step 3/3] Training model...")
    trainer, train_metrics, eval_metrics = trainer_obj.train(train_dataset, val_dataset)
    
    print("\n" + "="*70)
    print("Training Pipeline Complete!")
    print("="*70)
    print(f"\nFinal Validation Metrics:")
    print(f"  BLEU Score: {eval_metrics.get('eval_bleu', 0):.2f}")
    print(f"  ROUGE-1: {eval_metrics.get('eval_rouge1', 0):.4f}")
    print(f"  ROUGE-2: {eval_metrics.get('eval_rouge2', 0):.4f}")
    print(f"  ROUGE-L: {eval_metrics.get('eval_rougeL', 0):.4f}")
    print(f"\nModel saved to: {trainer_obj.output_dir}")


if __name__ == "__main__":
    main()
