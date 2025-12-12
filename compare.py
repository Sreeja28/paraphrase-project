"""
Comparison script: CPG vs LLM-based paraphrase generation
Tests with the provided cover letter passage
"""
import json
import time
from pathlib import Path
import pandas as pd

from config import TEST_SAMPLE, RESULTS_DIR, INPUT_CONSTRAINTS
from inference import ParaphraseGenerator
from llm_generator import LLMParaphraseGenerator
from evaluator import ParaphraseEvaluator


def compare_generators():
    """Compare CPG and LLM-based generators"""
    
    print("="*80)
    print(" Paraphrase Generation System - Comparative Evaluation")
    print("="*80)
    
    # Initialize components
    print("\n[1/4] Initializing generators and evaluator...")
    
    try:
        cpg = ParaphraseGenerator()
        cpg_available = True
        print(" CPG initialized")
    except Exception as e:
        print(f" CPG initialization failed: {e}")
        print("  Run training first: python train.py")
        cpg_available = False
    
    llm = LLMParaphraseGenerator()
    print(" LLM generator initialized")
    
    evaluator = ParaphraseEvaluator()
    print(" Evaluator initialized")
    
    # Prepare test sample
    print("\n[2/4] Preparing test sample...")
    print("-"*80)
    print(f"Test passage: Cover letter description")
    print(f"Word count: {len(TEST_SAMPLE.split())} words")
    print(f"Character count: {len(TEST_SAMPLE)} characters")
    
    # Validate constraints
    word_count = len(TEST_SAMPLE.split())
    min_words = INPUT_CONSTRAINTS["min_words"]
    max_words = INPUT_CONSTRAINTS["max_words"]
    
    if word_count < min_words or word_count > max_words:
        print(f"Warning: Test sample has {word_count} words")
        print(f"Expected: {min_words}-{max_words} words")
    else:
        print(f" Test sample meets length requirements ({min_words}-{max_words} words)")
    
    results = {}
    
    # Generate with CPG
    if cpg_available:
        print("\n[3/4] Generating paraphrase with CPG...")
        print("-"*80)
        
        cpg_result = cpg.generate_paraphrase(TEST_SAMPLE)
        
        print(f" CPG generation complete")
        print(f"  Output length: {cpg_result['output_word_count']} words")
        print(f"  Length ratio: {cpg_result['length_ratio']:.2%}")
        print(f"  Constraint met: {'Yes' if cpg_result['constraint_met'] else 'No'}")
        print(f"  Latency: {cpg_result['latency']:.3f} seconds")
        print(f"  Attempts: {cpg_result['attempts']}")
        
        results['cpg'] = cpg_result
    else:
        print("\n[3/4] Skipping CPG (not available)")
    
    # Generate with LLM
    print("\n[4/4] Generating paraphrase with LLM...")
    print("-"*80)
    
    llm_result = llm.generate_paraphrase(TEST_SAMPLE)
    
    print(f"✓ LLM generation complete")
    print(f"  Model: {llm_result['model']}")
    print(f"  Output length: {llm_result['output_word_count']} words")
    print(f"  Length ratio: {llm_result['length_ratio']:.2%}")
    print(f"  Constraint met: {'Yes' if llm_result['constraint_met'] else 'No'}")
    print(f"  Latency: {llm_result['latency']:.3f} seconds")
    
    results['llm'] = llm_result
    
    # Evaluate both
    print("\n" + "="*80)
    print(" EVALUATION RESULTS")
    print("="*80)
    
    evaluations = {}
    
    if cpg_available:
        print("\n" + "-"*80)
        print("CPG Evaluation")
        print("-"*80)
        
        cpg_metrics = evaluator.evaluate_single(
            TEST_SAMPLE,
            cpg_result['paraphrase'],
            cpg_result['latency']
        )
        evaluator.print_metrics(cpg_metrics, "CPG Metrics")
        evaluations['cpg'] = cpg_metrics
    
    print("\n" + "-"*80)
    print("LLM Evaluation")
    print("-"*80)
    
    llm_metrics = evaluator.evaluate_single(
        TEST_SAMPLE,
        llm_result['paraphrase'],
        llm_result['latency']
    )
    evaluator.print_metrics(llm_metrics, "LLM Metrics")
    evaluations['llm'] = llm_metrics
    
    # Comparison
    if cpg_available:
        print("\n" + "="*80)
        print(" COMPARATIVE ANALYSIS")
        print("="*80)
        
        comparison_metrics = [
            ("Length Ratio", "length_ratio"),
            ("Semantic Similarity", "semantic_similarity"),
            ("BERTScore F1", "bert_score_f1"),
            ("BLEU-4", "bleu_4"),
            ("ROUGE-L F1", "rouge_l_f"),
            ("Lexical Diversity", "lexical_diversity"),
            ("Latency (seconds)", "latency"),
        ]
        
        print(f"\n{'Metric':<30} {'CPG':>15} {'LLM':>15} {'Winner':>15}")
        print("-"*80)
        
        for metric_name, metric_key in comparison_metrics:
            cpg_val = cpg_metrics.get(metric_key, 0)
            llm_val = llm_metrics.get(metric_key, 0)
            
            if metric_key == "latency":
                winner = "CPG" if cpg_val < llm_val else "LLM"
            else:
                winner = "CPG" if cpg_val > llm_val else "LLM"
            
            if winner == "CPG" and abs(cpg_val - llm_val) < 0.001:
                winner = "Tie"
            elif winner == "LLM" and abs(cpg_val - llm_val) < 0.001:
                winner = "Tie"
            
            print(f"{metric_name:<30} {cpg_val:>15.4f} {llm_val:>15.4f} {winner:>15}")
    
    # Display paraphrases
    print("\n" + "="*80)
    print(" GENERATED PARAPHRASES")
    print("="*80)
    
    print("\n" + "-"*80)
    print("ORIGINAL TEXT:")
    print("-"*80)
    print(TEST_SAMPLE)
    
    if cpg_available:
        print("\n" + "-"*80)
        print("CPG PARAPHRASE:")
        print("-"*80)
        print(cpg_result['paraphrase'])
    
    print("\n" + "-"*80)
    print("LLM PARAPHRASE:")
    print("-"*80)
    print(llm_result['paraphrase'])
    
    # Save results
    output_file = RESULTS_DIR / "comparison_results.json"
    with open(output_file, 'w') as f:
        # Convert to serializable format
        save_data = {
            "test_sample": TEST_SAMPLE,
            "results": {},
            "evaluations": {}
        }
        
        for key, val in results.items():
            save_data["results"][key] = {
                k: (v if not isinstance(v, float) or not pd.isna(v) else 0)
                for k, v in val.items()
            }
        
        for key, val in evaluations.items():
            save_data["evaluations"][key] = {
                k: (float(v) if isinstance(v, (int, float)) and not pd.isna(v) else 0)
                for k, v in val.items()
            }
        
        json.dump(save_data, f, indent=2)
    
    print(f"\n Results saved to: {output_file}")
    
    # Create summary CSV
    summary_file = RESULTS_DIR / "comparison_summary.csv"
    
    summary_data = []
    for name, metrics in evaluations.items():
        row = {"model": name.upper()}
        row.update({
            k: (float(v) if isinstance(v, (int, float)) and not pd.isna(v) else 0)
            for k, v in metrics.items()
        })
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_file, index=False)
    
    print(f"✓ Summary saved to: {summary_file}")
    
    return results, evaluations


if __name__ == "__main__":
    # Download NLTK data
    import nltk
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass
    
    compare_generators()
