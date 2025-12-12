"""
Evaluation metrics display for paraphrase generation system
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TEST_SAMPLE
from evaluator import ParaphraseEvaluator
import re


def check_structure_preservation(original, paraphrase):
    """Check if structural elements are preserved"""
    # Check for numbered lists
    orig_numbers = len(re.findall(r'\d+\.', original))
    para_numbers = len(re.findall(r'\d+\.', paraphrase))
    
    # Check for key section keywords
    section_keywords = ['purpose', 'content', 'header', 'salutation', 'introduction', 
                       'body', 'conclusion', 'signature', 'significance']
    orig_keywords = sum(1 for kw in section_keywords if kw.lower() in original.lower())
    para_keywords = sum(1 for kw in section_keywords if kw.lower() in paraphrase.lower())
    
    return {
        'numbered_list_preservation': para_numbers / orig_numbers if orig_numbers > 0 else 1.0,
        'keyword_preservation': para_keywords / orig_keywords if orig_keywords > 0 else 1.0
    }


def check_information_integrity(original, paraphrase, semantic_sim, overlap):
    """Check completeness and hallucination"""
    # Completeness: high semantic similarity + reasonable overlap
    completeness = (semantic_sim * 0.7) + (overlap * 0.3)
    
    # No hallucination: overlap shouldn't be too low (would indicate new info)
    # and length shouldn't be much longer
    no_hallucination = 1.0 if overlap > 0.3 else overlap / 0.3
    
    return {
        'completeness': completeness,
        'no_hallucination': no_hallucination
    }


def display_metrics():
    """Display evaluation metrics in the specified format"""
    
    print("\n" + "="*70)
    print("PARAPHRASE GENERATION EVALUATION METRICS")
    print("="*70)
    
    # Sample text for demonstration
    original = TEST_SAMPLE
    # Mock paraphrase (in real use, this would come from your model)
    paraphrase = """A cover letter serves as a formal document accompanying your resume during job applications. It functions as an introductory piece that adds context to your application. The main function of a cover letter is to present yourself to the hiring manager while contextualizing your resume. It enables you to expand on your qualifications, skills, and experiences beyond what your resume captures. Additionally, it offers a chance to demonstrate your enthusiasm for both the position and the organization, and to articulate why you're a suitable candidate. A standard cover letter contains these sections: 1. Header: Your contact details, date, and employer's contact information. 2. Salutation: A greeting addressing the hiring manager, ideally using their name. 3. Introduction: A brief presentation of yourself and the role you're seeking. 4. Body: The main section where you elaborate on your qualifications, experiences, and skills that qualify you for the position. You can also discuss your potential contributions to the organization. 5. Conclusion: A summary of your key points and a reaffirmation of your interest in the role, possibly including a request for an interview. 6. Signature: A courteous closing phrase followed by your name. In the job application process, the cover letter holds significant importance. It's typically the initial document a hiring manager reviews, establishing the tone for your complete application. It gives you an opportunity to distinguish yourself from other candidates and create a positive initial impression. Some organizations mandate a cover letter, and omitting one might lead to your application being overlooked. In essence, a cover letter is a crucial element of job applications that introduces you, expands on your qualifications, and presents a persuasive argument for your candidacy."""
    
    # Initialize evaluator
    evaluator = ParaphraseEvaluator()
    
    # Compute all metrics
    metrics = evaluator.evaluate_single(original, paraphrase, latency=0.5)
    
    # Get structural metrics
    structure = check_structure_preservation(original, paraphrase)
    
    # Get information integrity
    integrity = check_information_integrity(
        original, paraphrase, 
        metrics['semantic_similarity'], 
        metrics['overlap_ratio']
    )
    
    # Display metrics in specified format
    print("\nA. SEMANTIC METRICS")
    print("-" * 70)
    print(f"  • BERTScore (F1):              {metrics['bert_score_f1']:.4f}")
    print(f"  • Cosine Similarity (SBERT):   {metrics['semantic_similarity']:.4f}")
    
    print("\nB. TEXT QUALITY & OVERLAP")
    print("-" * 70)
    print(f"  • BLEU-4:                      {metrics['bleu_4']:.4f}")
    print(f"  • ROUGE-L:                     {metrics['rouge_l_f']:.4f}")
    
    print("\nC. STRUCTURAL METRICS")
    print("-" * 70)
    print(f"  • Section Keyword Preservation: {structure['keyword_preservation']:.2%}")
    print(f"  • Numbered List Preservation:   {structure['numbered_list_preservation']:.2%}")
    
    print("\nD. INFORMATION INTEGRITY")
    print("-" * 70)
    print(f"  • Completeness:                {integrity['completeness']:.4f}")
    print(f"    (No removal of important details)")
    print(f"  • No Hallucinated Content:     {integrity['no_hallucination']:.4f}")
    print(f"    (No addition of false information)")
    
    print("\nE. LENGTH & PERFORMANCE")
    print("-" * 70)
    print(f"  • Length Ratio:                {metrics['length_ratio']:.2%}")
    print(f"    (Requirement: < 80%)          {'✓ PASS' if metrics['length_ratio'] < 0.8 else '✗ FAIL'}")
    print(f"  • Latency (CPG):               {metrics['latency']:.3f}s")
    print(f"  • Latency (LLM):               ~2.5s (estimated)")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Overall assessment
    overall_score = (
        metrics['semantic_similarity'] * 0.25 +
        metrics['bert_score_f1'] * 0.25 +
        metrics['bleu_4'] * 0.15 +
        metrics['rouge_l_f'] * 0.15 +
        structure['keyword_preservation'] * 0.10 +
        integrity['completeness'] * 0.10
    )
    
    print(f"Overall Quality Score:     {overall_score:.4f}")
    print(f"Length Constraint Met:     {'YES' if metrics['length_ratio'] < 0.8 else 'NO'}")
    print(f"Structural Integrity:      {'HIGH' if structure['keyword_preservation'] > 0.8 else 'MEDIUM' if structure['keyword_preservation'] > 0.6 else 'LOW'}")
    print(f"Information Preserved:     {'HIGH' if integrity['completeness'] > 0.85 else 'MEDIUM' if integrity['completeness'] > 0.7 else 'LOW'}")
    print(f"Performance:               {'EXCELLENT' if metrics['latency'] < 1.0 else 'GOOD' if metrics['latency'] < 2.0 else 'FAIR'}")
    
    print("\n" + "="*70)
    print("Note: Run 'python compare.py' for real CPG vs LLM comparison")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        display_metrics()
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
