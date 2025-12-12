"""
API wrapper for easy paraphrase generation
"""
from typing import Dict, List, Optional
from inference import ParaphraseGenerator
from llm_generator import LLMParaphraseGenerator
from evaluator import ParaphraseEvaluator


class ParaphraseAPI:
    """
    Simple API for paraphrase generation
    
    Usage:
        api = ParaphraseAPI()
        result = api.paraphrase("Your text here", method="cpg")
        print(result["paraphrase"])
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the API
        
        Args:
            model_path: Path to CPG model (optional)
        """
        self.cpg = None
        self.llm = None
        self.evaluator = ParaphraseEvaluator()
        
        # Try to load CPG
        try:
            self.cpg = ParaphraseGenerator(model_path=model_path)
            print("✓ CPG loaded")
        except Exception as e:
            print(f"⚠️  CPG not available: {e}")
        
        # Load LLM
        self.llm = LLMParaphraseGenerator()
        print("✓ LLM generator initialized")
    
    def paraphrase(
        self,
        text: str,
        method: str = "cpg",
        evaluate: bool = False,
        **kwargs
    ) -> Dict:
        """
        Generate paraphrase
        
        Args:
            text: Input text (200-400 words)
            method: "cpg" or "llm"
            evaluate: Whether to compute evaluation metrics
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with paraphrase and metadata
        """
        # Select generator
        if method.lower() == "cpg":
            if self.cpg is None:
                raise ValueError("CPG not available. Train model first or use method='llm'")
            generator = self.cpg
        elif method.lower() == "llm":
            generator = self.llm
        else:
            raise ValueError(f"Unknown method: {method}. Use 'cpg' or 'llm'")
        
        # Generate
        result = generator.generate_paraphrase(text, **kwargs)
        
        # Evaluate if requested
        if evaluate:
            metrics = self.evaluator.evaluate_single(
                text,
                result["paraphrase"],
                result.get("latency")
            )
            result["metrics"] = metrics
        
        return result
    
    def batch_paraphrase(
        self,
        texts: List[str],
        method: str = "cpg",
        evaluate: bool = False,
        **kwargs
    ) -> List[Dict]:
        """
        Generate paraphrases for multiple texts
        
        Args:
            texts: List of input texts
            method: "cpg" or "llm"
            evaluate: Whether to compute evaluation metrics
            **kwargs: Additional generation parameters
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, text in enumerate(texts):
            print(f"Processing {i+1}/{len(texts)}...")
            result = self.paraphrase(text, method=method, evaluate=evaluate, **kwargs)
            results.append(result)
        
        return results
    
    def compare(
        self,
        text: str,
        methods: List[str] = ["cpg", "llm"]
    ) -> Dict:
        """
        Compare multiple methods on the same text
        
        Args:
            text: Input text
            methods: List of methods to compare
            
        Returns:
            Dictionary with results for each method
        """
        comparison = {}
        
        for method in methods:
            try:
                result = self.paraphrase(text, method=method, evaluate=True)
                comparison[method] = result
            except Exception as e:
                print(f"Error with method {method}: {e}")
                comparison[method] = {"error": str(e)}
        
        return comparison


# Convenience functions
def paraphrase(text: str, method: str = "cpg") -> str:
    """
    Quick paraphrase function
    
    Args:
        text: Input text
        method: "cpg" or "llm"
        
    Returns:
        Paraphrased text
    """
    api = ParaphraseAPI()
    result = api.paraphrase(text, method=method)
    return result["paraphrase"]


def compare_methods(text: str) -> Dict:
    """
    Quick comparison function
    
    Args:
        text: Input text
        
    Returns:
        Comparison results
    """
    api = ParaphraseAPI()
    return api.compare(text)


# Example usage
if __name__ == "__main__":
    # Initialize API
    api = ParaphraseAPI()
    
    # Example text
    text = """
    A cover letter is a formal document that accompanies your resume when you 
    apply for a job. It serves as an introduction and provides additional 
    context for your application. The primary purpose of a cover letter is to 
    introduce yourself to the hiring manager and to provide context for your 
    resume. It allows you to elaborate on your qualifications, skills, and 
    experiences in a way that your resume may not fully capture.
    """
    
    print("="*70)
    print("Paraphrase API Demo")
    print("="*70)
    
    # Method 1: Simple paraphrase
    print("\n[1] Simple Paraphrase (CPG)")
    print("-"*70)
    result = api.paraphrase(text, method="cpg")
    print(f"Paraphrase: {result['paraphrase'][:200]}...")
    print(f"Length ratio: {result['length_ratio']:.2%}")
    print(f"Latency: {result['latency']:.3f}s")
    
    # Method 2: With evaluation
    print("\n[2] Paraphrase with Evaluation")
    print("-"*70)
    result = api.paraphrase(text, method="cpg", evaluate=True)
    print(f"BLEU-4: {result['metrics']['bleu_4']:.4f}")
    print(f"Semantic Similarity: {result['metrics']['semantic_similarity']:.4f}")
    
    # Method 3: Compare methods
    print("\n[3] Compare Methods")
    print("-"*70)
    comparison = api.compare(text)
    for method, result in comparison.items():
        if "error" not in result:
            print(f"\n{method.upper()}:")
            print(f"  Length ratio: {result['length_ratio']:.2%}")
            print(f"  Latency: {result['latency']:.3f}s")
            if "metrics" in result:
                print(f"  BLEU-4: {result['metrics']['bleu_4']:.4f}")
    
    print("\n" + "="*70)
    print("Demo complete!")
