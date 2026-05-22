# eval/run_full_eval.py
# Comprehensive Evaluation and Benchmark Suite for RAVEN-2 (100M parameter LLM)
# Runs Factual Knowledge Bench, Reasoning Bench, and Coherence/Degeneracy Bench.
# Outputs individual metrics, compiles a Markdown score table, and saves to results_{timestamp}.json.

import os
import sys
import time
import json
import argparse
import torch
from tokenizers import Tokenizer

# Add root folder to import paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from inference import load_model, clean_response
from eval.knowledge_bench import evaluate_knowledge
from eval.reasoning_bench import evaluate_reasoning

# 50 typical prompts to analyze generation coherence, loops, and degeneracy
COHERENCE_PROMPTS = [
    "Tell me a story about a little bird.",
    "Explain what computers do in one sentence.",
    "Describe the color blue to someone who is blind.",
    "Write a short haiku about night.",
    "What is the meaning of life?",
    "Why do leaves fall during autumn?",
    "Imagine a world without gravity. What happens?",
    "How does a refrigerator stay cold?",
    "Tell me a joke about robots.",
    "What is the best way to cook an egg?",
    "Write a paragraph about a futuristic city.",
    "Why do dogs bark?",
    "What is your favorite topic to discuss?",
    "Explain standard deviation to a five-year-old.",
    "What happens when ice melts?",
    "Describe a peaceful sunset.",
    "Who is your favorite character in literature?",
    "What makes a good friend?",
    "Write a short paragraph about the ocean.",
    "Is it better to be wise or intelligent?",
    "What causes waves in the ocean?",
    "Why is the sky blue?",
    "Tell me about the solar system.",
    "What is photosynthesis in your own words?",
    "Why do we sleep?",
    "What is the capital of Spain?",
    "Explain how a bicycle works.",
    "Write a short greeting for a king.",
    "What is standard model of physics?",
    "Describe the smell of rain on hot dust.",
    "Explain the concept of infinity.",
    "Why do cats purr?",
    "Explain the difference between weather and climate.",
    "What is standard deviation?",
    "Write a line about clock hands.",
    "Describe the taste of chocolate.",
    "Why do human beings build houses?",
    "What is the internet?",
    "Explain how trees grow.",
    "What is the purpose of books?",
    "Why is honey so sweet?",
    "Describe a crowded market.",
    "What is standard deviation in simple terms?",
    "What is standard deviation definition?",
    "How do airplanes fly?",
    "Write a haiku about rain.",
    "Tell me a story about a coding error.",
    "What is standard model of particles?",
    "Explain supply and demand theory.",
    "What is a solar eclipse simple explanation?"
]

def evaluate_coherence(model, tokenizer, max_new_tokens=80):
    """Evaluate coherence and check for loops, degeneracy, and length appropriateness.
    """
    model.eval()
    total_coherence = 0.0
    print("\n🚀 Running Coherence Bench (50 prompts)...")
    
    samples = []
    
    for idx, prompt in enumerate(COHERENCE_PROMPTS):
        formatted = f"<|system|>You are Raven. Sarcastic, dry, coherent.<|user|>{prompt}<|raven|>"
        input_ids = torch.tensor([tokenizer.encode(formatted).ids], dtype=torch.long, device=model.cos_freqs.device)
        
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids, 
                max_new_tokens=max_new_tokens,
                temperature=0.7, 
                top_k=40, 
                top_p=0.9, 
                repetition_penalty=1.1,
                use_kv_cache=True
            )
            
        output_text = tokenizer.decode(gen_ids[0].tolist()).strip()
        cleaned = clean_response(output_text)
        
        # 1. Coherence Scoring criteria:
        # A) Length check: should be between 10 and 300 characters
        length_score = 1.0 if (10 <= len(cleaned) <= 300) else 0.5
        if len(cleaned) == 0:
            length_score = 0.0
            
        # B) Repetition detection: Check if words loop
        words = cleaned.lower().split()
        repetition_score = 1.0
        if len(words) > 6:
            unique_ratio = len(set(words)) / len(words)
            # If unique words are less than 60%, it indicates possible looping
            if unique_ratio < 0.6:
                repetition_score = 0.5
            if unique_ratio < 0.4:
                repetition_score = 0.1
                
        # C) Gibberish/character loop check: Check if characters repeat sequentially
        gibberish_score = 1.0
        # Search for long repeating identical character sequences
        for char in set(cleaned):
            if cleaned.count(char * 5) > 0:
                gibberish_score = 0.2
                break
                
        # Average prompt coherence
        prompt_score = (length_score + repetition_score + gibberish_score) / 3.0
        total_coherence += prompt_score
        
        # Save sample outputs
        if len(samples) < 5:
            samples.append({
                "prompt": prompt,
                "response": cleaned,
                "score": f"{prompt_score:.2f} (len={length_score:.1f}, rep={repetition_score:.1f}, gib={gibberish_score:.1f})"
            })
            
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1:3d}/50] coherence checks completed")
            
    avg_coherence = (total_coherence / len(COHERENCE_PROMPTS)) * 100
    print("\n" + "=" * 40)
    print(" COHERENCE BENCHMARK RESULTS")
    print("=" * 40)
    print(f"  Average Coherence Accuracy: {avg_coherence:.2f}%")
    print("=" * 40)
    
    return avg_coherence, samples

def main():
    parser = argparse.ArgumentParser(description="RAVEN-2 Complete Evaluation Suite")
    parser.add_argument("--save-dir", type=str, default="eval", help="Directory to save JSON results")
    args = parser.parse_args()
    
    # Verify tokenizer exists
    if not os.path.exists(config.tokenizer_path):
        print(f"❌ Tokenizer not found at: {config.tokenizer_path}. Please train tokenizer first.")
        sys.exit(1)
        
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    
    # Load model weights
    model, checkpoint_type = load_model()
    
    print("\n" + "=" * 60)
    print(f" 🐦‍⬛  RAVEN-2 FULL EVALUATION PORTAL")
    print(f" Model size:   95.18M parameters")
    print(f" Checkpoint:   {checkpoint_type.upper()}")
    print(f" Running on:   {config.device.upper()}")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    
    # 1. Run Knowledge Bench
    k_scores, k_avg = evaluate_knowledge(model, tokenizer)
    
    # 2. Run Reasoning Bench
    r_scores, r_avg = evaluate_reasoning(model, tokenizer)
    
    # 3. Run Coherence Bench
    c_avg, c_samples = evaluate_coherence(model, tokenizer)
    
    total_time = time.time() - start_time
    
    # Compute compiled composite score
    composite_score = (k_avg + r_avg + c_avg) / 3.0
    
    # Display Gorgeous Markdown Scoreboard
    print("\n" + "=" * 60)
    print(" 🏆  RAVEN-2 COMPILATION SCOREBOARD")
    print("=" * 60)
    print(f"| Evaluation category  | Score    | Details                    |")
    print(f"|:---------------------|:---------|:---------------------------|")
    print(f"| Factual Knowledge    | {k_avg:7.2f}% | 100 prompts (Wikipedia)    |")
    print(f"| Reasoning & Logic    | {r_avg:7.2f}% | 50 math/puzzle/analogy     |")
    print(f"| Coherence & Syntax   | {c_avg:7.2f}% | 50 prompts loops check     |")
    print(f"|----------------------|----------|----------------------------|")
    print(f"| COMPOSITE SCORE      | {composite_score:7.2f}% | Weighted aggregate index   |")
    print("=" * 60)
    print(f"⚡ Full evaluation executed successfully in {total_time:.1f} seconds.\n")
    
    # Save results as a timestamped JSON file
    timestamp = int(time.time())
    results_path = os.path.join(args.save_dir, f"results_{timestamp}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    summary = {
        "timestamp": timestamp,
        "checkpoint": checkpoint_type,
        "device": config.device,
        "dtype": config.dtype_str,
        "evaluation_time_seconds": total_time,
        "scores": {
            "knowledge_bench": {
                "individual": k_scores,
                "average": k_avg
            },
            "reasoning_bench": {
                "individual": r_scores,
                "average": r_avg
            },
            "coherence_bench": {
                "average": c_avg,
                "samples": c_samples
            },
            "composite": composite_score
        }
    }
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"💾 Complete evaluation results logged in: {results_path}\n")

if __name__ == "__main__":
    main()
