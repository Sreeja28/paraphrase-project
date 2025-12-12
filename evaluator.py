"""
Evaluation module for paraphrase quality assessment
Computes multiple metrics: BLEU, ROUGE, semantic similarity, BERTScore, etc.
"""
import numpy as np
import time
from typing import List, Dict, Tuple
import re

# Metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class ParaphraseEvaluator:
    """Comprehensive paraphrase evaluation"""
    
    def __init__(self):
        print("Initializing ParaphraseEvaluator...")
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        # Initialize sentence transformer for semantic similarity
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Semantic similarity model loaded")
        except Exception as e:
            print(f"Warning: Could not load sentence transformer: {e}")
            self.sentence_model = None
        
        # Initialize BERTScore
        try:
            from bert_score import BERTScorer
            self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            print("✓ BERTScore model loaded")
        except Exception as e:
            print(f"Warning: Could not load BERTScore: {e}")
            self.bert_scorer = None
        
        # BLEU smoothing
        self.smoothing = SmoothingFunction()
        
        print("✓ Evaluator initialized")
    
    def count_words(self, text):
        """Count words in text"""
        return len(re.findall(r'\b\w+\b', text))
    
    def compute_bleu(self, reference: str, candidate: str) -> Dict[str, float]:
        """Compute BLEU scores"""
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()
        
        # BLEU-1 to BLEU-4
        bleu_1 = sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=self.smoothing.method1
        )
        
        bleu_2 = sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=self.smoothing.method1
        )
        
        bleu_3 = sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=(0.33, 0.33, 0.33, 0),
            smoothing_function=self.smoothing.method1
        )
        
        bleu_4 = sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.smoothing.method1
        )
        
        return {
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "bleu_3": bleu_3,
            "bleu_4": bleu_4,
        }
    
    def compute_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Compute ROUGE scores"""
        scores = self.rouge_scorer.score(reference, candidate)
        
        return {
            "rouge_1_f": scores['rouge1'].fmeasure,
            "rouge_1_p": scores['rouge1'].precision,
            "rouge_1_r": scores['rouge1'].recall,
            "rouge_2_f": scores['rouge2'].fmeasure,
            "rouge_2_p": scores['rouge2'].precision,
            "rouge_2_r": scores['rouge2'].recall,
            "rouge_l_f": scores['rougeL'].fmeasure,
            "rouge_l_p": scores['rougeL'].precision,
            "rouge_l_r": scores['rougeL'].recall,
        }
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using sentence embeddings"""
        if not self.sentence_model:
            return 0.0
        
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error computing semantic similarity: {e}")
            return 0.0
    
    def compute_bert_score(self, reference: str, candidate: str) -> Dict[str, float]:
        """Compute BERTScore"""
        if not self.bert_scorer:
            return {"bert_score_f1": 0.0, "bert_score_precision": 0.0, "bert_score_recall": 0.0}
        
        try:
            P, R, F1 = self.bert_scorer.score([candidate], [reference])
            return {
                "bert_score_f1": float(F1[0]),
                "bert_score_precision": float(P[0]),
                "bert_score_recall": float(R[0]),
            }
        except Exception as e:
            print(f"Error computing BERTScore: {e}")
            return {"bert_score_f1": 0.0, "bert_score_precision": 0.0, "bert_score_recall": 0.0}
    
    def compute_lexical_diversity(self, text: str) -> float:
        """Compute lexical diversity (unique words / total words)"""
        words = text.lower().split()
        if len(words) == 0:
            return 0.0
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def compute_overlap_ratio(self, text1: str, text2: str) -> float:
        """Compute word overlap ratio between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if len(union) > 0 else 0.0
    
    def evaluate_single(
        self,
        original: str,
        paraphrase: str,
        latency: float = None
    ) -> Dict:
        """
        Comprehensive evaluation of a single paraphrase
        
        Args:
            original: Original text
            paraphrase: Paraphrased text
            latency: Generation latency (optional)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Word counts
        original_word_count = self.count_words(original)
        paraphrase_word_count = self.count_words(paraphrase)
        length_ratio = paraphrase_word_count / original_word_count if original_word_count > 0 else 0
        
        # BLEU scores
        bleu_scores = self.compute_bleu(original, paraphrase)
        
        # ROUGE scores
        rouge_scores = self.compute_rouge(original, paraphrase)
        
        # Semantic similarity
        semantic_sim = self.compute_semantic_similarity(original, paraphrase)
        
        # BERTScore
        bert_scores = self.compute_bert_score(original, paraphrase)
        
        # Lexical diversity
        lexical_diversity = self.compute_lexical_diversity(paraphrase)
        
        # Overlap ratio
        overlap_ratio = self.compute_overlap_ratio(original, paraphrase)
        
        # Combine all metrics
        metrics = {
            "original_word_count": original_word_count,
            "paraphrase_word_count": paraphrase_word_count,
            "length_ratio": length_ratio,
            "latency": latency,
            "semantic_similarity": semantic_sim,
            "lexical_diversity": lexical_diversity,
            "overlap_ratio": overlap_ratio,
            **bleu_scores,
            **rouge_scores,
            **bert_scores,
        }
        
        return metrics
    
    def evaluate_batch(
        self,
        originals: List[str],
        paraphrases: List[str],
        latencies: List[float] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Evaluate multiple paraphrases
        
        Returns:
            Tuple of (individual_results, aggregate_metrics)
        """
        if latencies is None:
            latencies = [None] * len(originals)
        
        results = []
        for orig, para, lat in zip(originals, paraphrases, latencies):
            result = self.evaluate_single(orig, para, lat)
            results.append(result)
        
        # Compute aggregate statistics
        aggregate = self._compute_aggregate(results)
        
        return results, aggregate
    
    def _compute_aggregate(self, results: List[Dict]) -> Dict:
        """Compute aggregate statistics from individual results"""
        if not results:
            return {}
        
        aggregate = {}
        
        # Get all numeric keys
        numeric_keys = [k for k in results[0].keys() if isinstance(results[0][k], (int, float))]
        
        for key in numeric_keys:
            values = [r[key] for r in results if r[key] is not None]
            if values:
                aggregate[f"{key}_mean"] = np.mean(values)
                aggregate[f"{key}_std"] = np.std(values)
                aggregate[f"{key}_min"] = np.min(values)
                aggregate[f"{key}_max"] = np.max(values)
        
        return aggregate
    
    def print_metrics(self, metrics: Dict, title: str = "Evaluation Metrics"):
        """Pretty print metrics"""
        print(f"\n{'='*70}")
        print(title)
        print("="*70)
        
        # Group metrics
        length_metrics = ["original_word_count", "paraphrase_word_count", "length_ratio"]
        quality_metrics = ["semantic_similarity", "bert_score_f1", "bleu_4", "rouge_l_f"]
        diversity_metrics = ["lexical_diversity", "overlap_ratio"]
        performance_metrics = ["latency"]
        
        def print_group(group_name, keys):
            print(f"\n{group_name}:")
            for key in keys:
                if key in metrics and metrics[key] is not None:
                    value = metrics[key]
                    if isinstance(value, float):
                        if "ratio" in key or "similarity" in key or "diversity" in key:
                            print(f"  {key:30s}: {value:.4f}")
                        else:
                            print(f"  {key:30s}: {value:.4f}")
                    else:
                        print(f"  {key:30s}: {value}")
        
        print_group("Length Metrics", length_metrics)
        print_group("Quality Metrics", quality_metrics)
        print_group("Diversity Metrics", diversity_metrics)
        print_group("Performance Metrics", performance_metrics)
        
        # Print all BLEU scores
        print("\nBLEU Scores:")
        for i in range(1, 5):
            key = f"bleu_{i}"
            if key in metrics:
                print(f"  {key:30s}: {metrics[key]:.4f}")
        
        # Print all ROUGE scores
        print("\nROUGE Scores:")
        for rouge_type in ["rouge_1", "rouge_2", "rouge_l"]:
            for metric_type in ["f", "p", "r"]:
                key = f"{rouge_type}_{metric_type}"
                if key in metrics:
                    print(f"  {key:30s}: {metrics[key]:.4f}")


def main():
    """Test the evaluator"""
    print("Testing ParaphraseEvaluator...")
    
    evaluator = ParaphraseEvaluator()
    
    # Test texts
    original = "Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data."
    paraphrase1 = "A branch of AI called machine learning concentrates on developing systems capable of learning from information."
    paraphrase2 = "Machine learning, part of AI, is about creating systems that learn from data."
    
    # Evaluate
    print("\n" + "="*70)
    print("Paraphrase 1 Evaluation")
    print("="*70)
    metrics1 = evaluator.evaluate_single(original, paraphrase1, latency=0.5)
    evaluator.print_metrics(metrics1)
    
    print("\n" + "="*70)
    print("Paraphrase 2 Evaluation")
    print("="*70)
    metrics2 = evaluator.evaluate_single(original, paraphrase2, latency=0.3)
    evaluator.print_metrics(metrics2)


if __name__ == "__main__":
    # Download required NLTK data
    import nltk
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass
    
    main()
