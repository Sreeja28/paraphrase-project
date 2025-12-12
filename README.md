# Paraphrase-project


This repository worked on Paraphrase Generation System for paragraph-level paraphrasing (designed for inputs of 200–400 words) by implementing a Custom Paraphrase Generator (CPG) fine-tuned from T5 and a comparison pipeline against an LLM-based paraphrase generator (OpenAI gpt-3.5). The system fine-tunes a seq2seq model on multiple paraphrase datasets (PAWS, Quora, MRPC, ParaNMT), enforces a length constraint on outputs (target ≥ 80% of the input paragraph), and evaluates generated paraphrases using BLEU, ROUGE, BERTScore, SBERT semantic similarity, lexical-diversity/overlap metrics and latency. The repo includes training, inference, evaluation, visualization, and a ready-to-run comparison.

##  Run Order Summary

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

