"""
Configuration file for Paraphrase Generation System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_CONFIG = {
    "base_model": "t5-small",  # Smaller model for faster training (60M params)
    "max_input_length": 512,  # tokens for 200-400 words
    "max_output_length": 400,  # 78% of 512 to get closer to target
    "target_length_ratio": 0.75,  # Target 75% for better semantic preservation
    "max_length_ratio": 0.79,  # Strictly less than 80%
    "num_beams": 4,  # Balance between quality and speed
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
}

# Training configurations (OPTIMIZED FOR BETTER PERFORMANCE)
TRAINING_CONFIG = {
    "learning_rate": 3e-5,  # Lower LR for better convergence
    "batch_size": 32,  # Larger batch for faster training
    "num_epochs": 100,  # Increased to 10 epochs for much better learning
    "warmup_steps": 200,  # More warmup for stability
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 2,  # Accumulate for effective batch of 32
    "fp16": True,
    "save_steps": 1000,  # More frequent saves
    "eval_steps": 500,  # More frequent evaluation
    "logging_steps": 50,
    "max_train_samples": 15000,  # Increased training data
}

# Input text constraints
INPUT_CONSTRAINTS = {
    "min_words": 200,
    "max_words": 400,
    "max_output_ratio": 0.79,  # Output must be LESS than 80% of input length
    "target_output_ratio": 0.75,  # Target 75% for better balance
    "min_output_ratio": 0.65,  # Minimum 65% to preserve semantic content
}

# Evaluation metrics
EVALUATION_METRICS = [
    "bleu",
    "rouge",
    "semantic_similarity",
    "bert_score",
    "length_ratio",
    "latency",
]

# LLM API configuration (for comparison)
LLM_CONFIG = {
    "model": "gpt-3.5-turbo",  # Can be changed to gpt-4
    "temperature": 0.7,
    "max_tokens": 600,
}

# Test sample
TEST_SAMPLE = """A cover letter is a formal document that accompanies your resume when you apply for a job. It serves as an introduction and provides additional context for your application. Here's a breakdown of its various aspects: Purpose The primary purpose of a cover letter is to introduce yourself to the hiring manager and to provide context for your resume. It allows you to elaborate on your qualifications, skills, and experiences in a way that your resume may not fully capture. It's also an opportunity to express your enthusiasm for the role and the company, and to explain why you would be a good fit. Content A typical cover letter includes the following sections:  
1. Header: Includes your contact information, the date, and the employer's contact information. 
2. Salutation: A greeting to the hiring manager, preferably personalized with their name. 
3. Introduction: Briefly introduces who you are and the position you're applying for. 
4. Body: This is the core of your cover letter where you discuss your qualifications, experiences, and skills that make you suitable for the job. You can also mention how you can contribute to the company. 
5. Conclusion: Summarizes your points and reiterates your enthusiasm for the role. You can also include a call to action, like asking for an interview
6. Signature: A polite closing ("Sincerely," "Best regards," etc.) followed by your name. Significance in the Job Application Process The cover letter is often the first document that a hiring manager will read, so it sets the tone for your entire application. It provides you with a chance to stand out among other applicants and to make a strong first impression. Some employers specifically require a cover letter, and failing to include one could result in your application being disregarded. In summary, a cover letter is an essential component of a job application that serves to introduce you, elaborate on your qualifications, and make a compelling case for why you should be considered for the position."""
