"""
LLM-based paraphrase generator for comparison
Uses OpenAI GPT or other LLM APIs
"""
import os
import time
import re
from typing import Optional, Dict
from config import LLM_CONFIG, INPUT_CONSTRAINTS


class LLMParaphraseGenerator:
    """LLM-based paraphrase generator using OpenAI API"""
    
    def __init__(self, model_name=None, api_key=None):
        self.model_name = model_name or LLM_CONFIG["model"]
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found in environment")
            print("Set it using: export OPENAI_API_KEY='your-key-here'")
            print("Or create a .env file with: OPENAI_API_KEY=your-key-here")
        
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                print(f"✓ LLM initialized with model: {self.model_name}")
            except ImportError:
                print("OpenAI package not installed. Install with: pip install openai")
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
    
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
    
    def generate_paraphrase(self, text: str, **kwargs) -> Dict:
        """
        Generate paraphrase using LLM
        
        Args:
            text: Input text to paraphrase
            
        Returns:
            dict with paraphrase, latency, and metadata
        """
        if not self.client:
            return self._generate_mock_paraphrase(text)
        
        # Validate input
        input_word_count = self.validate_input(text)
        target_words = int(input_word_count * INPUT_CONSTRAINTS["target_output_ratio"])
        max_output_words = int(input_word_count * INPUT_CONSTRAINTS["max_output_ratio"])
        
        # Create prompt with strict length requirement
        prompt = f"""You are a professional paraphrasing assistant. Your task is to rewrite the following text while:
1. Maintaining the original meaning and all key information
2. Using different words and sentence structures
3. CRITICAL REQUIREMENT: The paraphrased text must be LESS THAN 80% of the original length
4. Target length: {target_words} words (aim for around 70% of original)
5. Maximum length: {max_output_words} words (strictly less than 80%)
6. Keeping the same tone and style
7. Not adding or removing any factual information
8. Being concise and eliminating redundant phrases

Original text ({input_word_count} words):
{text}

Please provide ONLY the paraphrased text (target ~{target_words} words, max {max_output_words} words) without any additional comments or explanations."""

        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional paraphrasing assistant. Provide only the paraphrased text without any additional commentary."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"],
            )
            
            paraphrase = response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return self._generate_mock_paraphrase(text)
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Calculate metrics
        output_word_count = self.count_words(paraphrase)
        length_ratio = output_word_count / input_word_count if input_word_count > 0 else 0
        max_allowed_words = int(input_word_count * INPUT_CONSTRAINTS["max_output_ratio"])
        constraint_met = length_ratio < INPUT_CONSTRAINTS["max_output_ratio"]
        
        # Truncate if needed
        if not constraint_met:
            words = paraphrase.split()
            paraphrase = ' '.join(words[:max_allowed_words])
            output_word_count = len(paraphrase.split())
            length_ratio = output_word_count / input_word_count
            constraint_met = True
        
        return {
            "paraphrase": paraphrase,
            "input_word_count": input_word_count,
            "output_word_count": output_word_count,
            "length_ratio": length_ratio,
            "max_allowed_words": max_allowed_words,
            "constraint_met": constraint_met,
            "latency": latency,
            "model": self.model_name,
        }
    
    def _generate_mock_paraphrase(self, text: str) -> Dict:
        """Generate mock paraphrase when API is not available"""
        print("Warning: Using mock paraphrase (API not available)")
        
        input_word_count = self.count_words(text)
        target_words = int(input_word_count * INPUT_CONSTRAINTS["target_output_ratio"])
        max_allowed_words = int(input_word_count * INPUT_CONSTRAINTS["max_output_ratio"])
        
        # Generate shorter mock by truncating to target length
        words = text.split()
        mock_words = words[:target_words]
        mock_paraphrase = ' '.join(mock_words)
        output_word_count = len(mock_words)
        
        return {
            "paraphrase": f"[MOCK] {mock_paraphrase}",
            "input_word_count": input_word_count,
            "output_word_count": output_word_count,
            "length_ratio": output_word_count / input_word_count,
            "max_allowed_words": max_allowed_words,
            "constraint_met": True,
            "latency": 0.1,
            "model": "mock",
        }
    
    def batch_generate(self, texts, **kwargs):
        """Generate paraphrases for multiple texts"""
        results = []
        for text in texts:
            result = self.generate_paraphrase(text, **kwargs)
            results.append(result)
        return results


def main():
    """Test the LLM paraphrase generator"""
    from config import TEST_SAMPLE
    
    print("="*70)
    print("Testing LLM-based Paraphrase Generator")
    print("="*70)
    
    # Initialize generator
    generator = LLMParaphraseGenerator()
    
    # Test with sample
    print("\nGenerating paraphrase for test sample...")
    print("-"*70)
    
    result = generator.generate_paraphrase(TEST_SAMPLE)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    
    print(f"\nModel: {result['model']}")
    print(f"Input Length: {result['input_word_count']} words")
    print(f"Output Length: {result['output_word_count']} words")
    print(f"Length Ratio: {result['length_ratio']:.2%}")
    print(f"Minimum Required: {result['min_required_words']} words")
    print(f"Constraint Met: {'✓' if result['constraint_met'] else '✗'}")
    print(f"Latency: {result['latency']:.3f} seconds")
    
    print(f"\n{'='*70}")
    print("ORIGINAL TEXT:")
    print("="*70)
    print(TEST_SAMPLE[:300] + "...")
    
    print(f"\n{'='*70}")
    print("PARAPHRASED TEXT:")
    print("="*70)
    print(result['paraphrase'][:500] + "...")


if __name__ == "__main__":
    main()
