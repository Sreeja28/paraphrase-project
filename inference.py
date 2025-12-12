"""
Inference module for Custom Paraphrase Generator (CPG)
Generates paraphrases with length constraints
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import MODEL_CONFIG, INPUT_CONSTRAINTS, MODELS_DIR
import time
import re


class ParaphraseGenerator:
    """Custom Paraphrase Generator with length constraints"""
    
    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or str(MODELS_DIR / "cpg_model")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading CPG model from {self.model_path}")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Model loaded successfully")
        
    def count_words(self, text):
        """Count words in text"""
        return len(re.findall(r'\b\w+\b', text))
    
    def validate_input(self, text):
        """Validate input text meets constraints"""
        word_count = self.count_words(text)
        min_words = INPUT_CONSTRAINTS["min_words"]
        max_words = INPUT_CONSTRAINTS["max_words"]
        
        if word_count < min_words:
            raise ValueError(
                f"Input text has {word_count} words. "
                f"Minimum required: {min_words} words"
            )
        
        if word_count > max_words:
            raise ValueError(
                f"Input text has {word_count} words. "
                f"Maximum allowed: {max_words} words"
            )
        
        return word_count
    
    def generate_paraphrase(
        self,
        text,
        num_beams=None,
        temperature=None,
        top_p=None,
        repetition_penalty=None,
        length_penalty=1.0,
        max_attempts=5
    ):
        """
        Generate paraphrase with length constraints
        
        Args:
            text: Input text to paraphrase
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repetition
            length_penalty: Penalty for length (>1.0 encourages longer sequences)
            max_attempts: Maximum attempts to meet length constraint
            
        Returns:
            dict with paraphrase, latency, and metadata
        """
        # Validate input
        input_word_count = self.validate_input(text)
        target_output_words = int(input_word_count * INPUT_CONSTRAINTS["target_output_ratio"])
        max_output_words = int(input_word_count * INPUT_CONSTRAINTS["max_output_ratio"])
        
        # Use config defaults if not specified
        num_beams = num_beams or MODEL_CONFIG["num_beams"]
        temperature = temperature or MODEL_CONFIG["temperature"]
        top_p = top_p or MODEL_CONFIG["top_p"]
        repetition_penalty = repetition_penalty or MODEL_CONFIG["repetition_penalty"]
        
        # Prepare input
        if "t5" in self.model_path.lower():
            input_text = f"paraphrase: {text}"
        else:
            input_text = text
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=MODEL_CONFIG["max_input_length"],
            truncation=True,
            padding=True
        ).to(self.device)
        
        input_length = inputs["input_ids"].shape[1]
        
        # Calculate target and max output length (< 80% of input length in tokens)
        target_output_length = int(input_length * INPUT_CONSTRAINTS["target_output_ratio"])
        max_output_length = int(input_length * INPUT_CONSTRAINTS["max_output_ratio"])
        min_output_length = int(input_length * INPUT_CONSTRAINTS.get("min_output_ratio", 0.65))
        
        start_time = time.time()
        
        # Generate with multiple attempts if needed
        best_paraphrase = None
        best_word_count = 0
        best_ratio = float('inf')
        
        for attempt in range(max_attempts):
            # Fine-tune length_penalty for 65-75% range
            # Start near neutral (1.0) and adjust slightly
            current_length_penalty = 0.8 - (attempt * 0.1)
            current_max_length = min(MODEL_CONFIG["max_output_length"], max_output_length * 2)  # Token overhead
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=current_max_length,
                    min_length=min_output_length,  # Enforce minimum for semantic preservation
                    num_beams=num_beams,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    length_penalty=current_length_penalty,
                    early_stopping=True,
                    do_sample=False,  # Use beam search for quality
                    no_repeat_ngram_size=2,  # Reduced for more flexibility
                    num_return_sequences=1,
                )
            
            # Decode output
            paraphrase = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_word_count = self.count_words(paraphrase)
            current_ratio = output_word_count / input_word_count
            
            # Check if length constraint is met (65-79%)
            min_ratio = INPUT_CONSTRAINTS.get("min_output_ratio", 0.65)
            max_ratio = INPUT_CONSTRAINTS["max_output_ratio"]
            
            if min_ratio <= current_ratio < max_ratio:
                best_paraphrase = paraphrase
                best_word_count = output_word_count
                best_ratio = current_ratio
                break
            
            # Keep track of best attempt (closest to target within range)
            if current_ratio < max_ratio:
                if best_ratio == float('inf') or abs(current_ratio - INPUT_CONSTRAINTS["target_output_ratio"]) < abs(best_ratio - INPUT_CONSTRAINTS["target_output_ratio"]):
                    best_paraphrase = paraphrase
                    best_word_count = output_word_count
                    best_ratio = current_ratio
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Calculate metrics
        length_ratio = best_word_count / input_word_count if input_word_count > 0 else 0
        max_allowed_words = int(input_word_count * INPUT_CONSTRAINTS["max_output_ratio"])
        min_required_words = int(input_word_count * INPUT_CONSTRAINTS.get("min_output_ratio", 0.65))
        constraint_met = length_ratio < INPUT_CONSTRAINTS["max_output_ratio"]
        
        # If still above limit, truncate ONLY if necessary (preserve quality)
        if not constraint_met and best_paraphrase:
            # Smart truncation: try to end at sentence boundary
            words = best_paraphrase.split()
            truncated_words = words[:max_allowed_words]
            truncated_text = ' '.join(truncated_words)
            
            # Try to end at a sentence boundary
            if '.' in truncated_text:
                sentences = truncated_text.split('.')
                truncated_text = '.'.join(sentences[:-1]) + '.'
            
            best_paraphrase = truncated_text
            best_word_count = len(best_paraphrase.split())
            length_ratio = best_word_count / input_word_count
            constraint_met = True
        
        return {
            "paraphrase": best_paraphrase,
            "input_word_count": input_word_count,
            "output_word_count": best_word_count,
            "length_ratio": length_ratio,
            "max_allowed_words": max_allowed_words,
            "min_required_words": min_required_words,
            "constraint_met": constraint_met,
            "latency": latency,
            "attempts": attempt + 1,
        }
    
    def batch_generate(self, texts, **kwargs):
        """Generate paraphrases for multiple texts"""
        results = []
        for text in texts:
            result = self.generate_paraphrase(text, **kwargs)
            results.append(result)
        return results


def main():
    """Test the paraphrase generator"""
    from config import TEST_SAMPLE
    
    print("="*70)
    print("Testing Custom Paraphrase Generator (CPG)")
    print("="*70)
    
    # Initialize generator
    generator = ParaphraseGenerator()
    
    # Test with sample
    print("\nGenerating paraphrase for test sample...")
    print("-"*70)
    
    result = generator.generate_paraphrase(TEST_SAMPLE)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    
    print(f"\nInput Length: {result['input_word_count']} words")
    print(f"Output Length: {result['output_word_count']} words")
    print(f"Length Ratio: {result['length_ratio']:.2%}")
    print(f"Minimum Required: {result['min_required_words']} words")
    print(f"Constraint Met: {'✓' if result['constraint_met'] else '✗'}")
    print(f"Latency: {result['latency']:.3f} seconds")
    print(f"Attempts: {result['attempts']}")
    
    print(f"\n{'='*70}")
    print("ORIGINAL TEXT:")
    print("="*70)
    print(TEST_SAMPLE[:300] + "...")
    
    print(f"\n{'='*70}")
    print("PARAPHRASED TEXT:")
    print("="*70)
    print(result['paraphrase'])


if __name__ == "__main__":
    main()
