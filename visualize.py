"""
Visualization module for evaluation results
Creates plots comparing CPG and LLM performance
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import RESULTS_DIR, PLOTS_DIR


def load_results():
    """Load comparison results"""
    results_file = RESULTS_DIR / "comparison_results.json"
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run compare.py first to generate results")
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return data


def plot_comparison(data):
    """Create comparison plots"""
    
    if not data or 'evaluations' not in data:
        print("No evaluation data found")
        return
    
    evaluations = data['evaluations']
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CPG vs LLM Paraphrase Generation Comparison', fontsize=16, fontweight='bold')
    
    # Prepare data
    models = list(evaluations.keys())
    model_names = [m.upper() for m in models]
    
    # 1. Quality Metrics (BLEU, ROUGE, BERTScore)
    ax = axes[0, 0]
    quality_metrics = ['bleu_4', 'rouge_l_f', 'bert_score_f1']
    quality_labels = ['BLEU-4', 'ROUGE-L F1', 'BERTScore F1']
    
    x = range(len(quality_labels))
    width = 0.35
    
    for i, model in enumerate(models):
        values = [evaluations[model].get(m, 0) for m in quality_metrics]
        ax.bar([p + width*i for p in x], values, width, label=model.upper())
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Quality Metrics Comparison')
    ax.set_xticks([p + width/2 for p in x])
    ax.set_xticklabels(quality_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. BLEU Scores (1-4)
    ax = axes[0, 1]
    bleu_metrics = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']
    
    for model in models:
        values = [evaluations[model].get(m, 0) for m in bleu_metrics]
        ax.plot(range(1, 5), values, marker='o', label=model.upper(), linewidth=2)
    
    ax.set_xlabel('BLEU-N')
    ax.set_ylabel('Score')
    ax.set_title('BLEU Scores (N-gram)')
    ax.set_xticks(range(1, 5))
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. ROUGE Scores
    ax = axes[0, 2]
    rouge_metrics = ['rouge_1_f', 'rouge_2_f', 'rouge_l_f']
    rouge_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    
    x = range(len(rouge_labels))
    for i, model in enumerate(models):
        values = [evaluations[model].get(m, 0) for m in rouge_metrics]
        ax.bar([p + width*i for p in x], values, width, label=model.upper())
    
    ax.set_xlabel('ROUGE Variants')
    ax.set_ylabel('F1 Score')
    ax.set_title('ROUGE Scores Comparison')
    ax.set_xticks([p + width/2 for p in x])
    ax.set_xticklabels(rouge_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Semantic Similarity & Diversity
    ax = axes[1, 0]
    sim_metrics = ['semantic_similarity', 'lexical_diversity']
    sim_labels = ['Semantic\nSimilarity', 'Lexical\nDiversity']
    
    x = range(len(sim_labels))
    for i, model in enumerate(models):
        values = [evaluations[model].get(m, 0) for m in sim_metrics]
        ax.bar([p + width*i for p in x], values, width, label=model.upper())
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Similarity & Diversity')
    ax.set_xticks([p + width/2 for p in x])
    ax.set_xticklabels(sim_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 5. Length Metrics
    ax = axes[1, 1]
    length_data = []
    for model in models:
        length_data.append({
            'Model': model.upper(),
            'Original': evaluations[model].get('original_word_count', 0),
            'Generated': evaluations[model].get('paraphrase_word_count', 0)
        })
    
    df_length = pd.DataFrame(length_data)
    x = range(len(models))
    
    ax.bar([p - width/2 for p in x], df_length['Original'], width, label='Original', alpha=0.7)
    ax.bar([p + width/2 for p in x], df_length['Generated'], width, label='Generated', alpha=0.7)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Word Count')
    ax.set_title('Length Comparison (Word Count)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add 80% threshold line
    min_required = int(evaluations[models[0]].get('original_word_count', 0) * 0.8)
    ax.axhline(y=min_required, color='r', linestyle='--', linewidth=2, label='80% Threshold')
    ax.legend()
    
    # 6. Latency Comparison
    ax = axes[1, 2]
    latencies = [evaluations[model].get('latency', 0) for model in models]
    colors = ['#2ecc71' if l < 1 else '#e74c3c' for l in latencies]
    
    bars = ax.bar(model_names, latencies, color=colors, alpha=0.7)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Generation Latency')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, lat in zip(bars, latencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{lat:.3f}s',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = PLOTS_DIR / "comparison_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to: {plot_file}")
    
    plt.close()
    
    # Create individual metric comparison
    create_radar_chart(evaluations, models)


def create_radar_chart(evaluations, models):
    """Create radar chart for overall comparison"""
    
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Select key metrics (normalized)
    metrics = [
        ('bleu_4', 'BLEU-4'),
        ('rouge_l_f', 'ROUGE-L'),
        ('bert_score_f1', 'BERTScore'),
        ('semantic_similarity', 'Semantic Sim'),
        ('lexical_diversity', 'Lexical Div'),
    ]
    
    # Normalize latency (invert so lower is better -> higher score)
    max_latency = max([evaluations[m].get('latency', 1) for m in models])
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[1] for m in metrics])
    
    for model in models:
        values = [evaluations[model].get(m[0], 0) for m in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model.upper())
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance Comparison', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plot_file = PLOTS_DIR / "radar_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Radar chart saved to: {plot_file}")
    
    plt.close()


def create_summary_table():
    """Create a summary comparison table"""
    
    results_file = RESULTS_DIR / "comparison_results.json"
    
    if not results_file.exists():
        print("Results file not found")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    evaluations = data['evaluations']
    
    # Create comparison dataframe
    metrics_to_compare = [
        ('Length Ratio', 'length_ratio', '{:.2%}'),
        ('Semantic Similarity', 'semantic_similarity', '{:.4f}'),
        ('BERTScore F1', 'bert_score_f1', '{:.4f}'),
        ('BLEU-4', 'bleu_4', '{:.4f}'),
        ('ROUGE-L F1', 'rouge_l_f', '{:.4f}'),
        ('Lexical Diversity', 'lexical_diversity', '{:.4f}'),
        ('Latency (s)', 'latency', '{:.3f}'),
    ]
    
    table_data = []
    for display_name, metric_key, fmt in metrics_to_compare:
        row = {'Metric': display_name}
        for model in evaluations.keys():
            value = evaluations[model].get(metric_key, 0)
            row[model.upper()] = fmt.format(value)
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    table_file = RESULTS_DIR / "metrics_comparison_table.csv"
    df.to_csv(table_file, index=False)
    print(f"✓ Comparison table saved to: {table_file}")
    
    # Print table
    print("\n" + "="*70)
    print("METRICS COMPARISON TABLE")
    print("="*70)
    print(df.to_string(index=False))


def main():
    """Generate all visualizations"""
    
    print("="*70)
    print("Generating Visualization and Analysis")
    print("="*70)
    
    # Load results
    print("\nLoading results...")
    data = load_results()
    
    if data is None:
        return
    
    print("✓ Results loaded")
    
    # Create plots
    print("\nGenerating plots...")
    plot_comparison(data)
    
    # Create summary table
    print("\nGenerating summary table...")
    create_summary_table()
    
    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)
    print(f"Check the '{PLOTS_DIR}' directory for plots")
    print(f"Check the '{RESULTS_DIR}' directory for tables")


if __name__ == "__main__":
    main()
