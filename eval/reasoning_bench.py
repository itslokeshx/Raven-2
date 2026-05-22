# eval/reasoning_bench.py
# Reasoning and Logic Benchmark for RAVEN-2 (100M parameter LLM)
# Contains 50 challenging reasoning prompts across 3 categories: Mathematics, Logic Puzzles, and Analogies.
# Evaluates precise correctness using exact value check or numerical overlap metrics.

import os
import sys

# Arithmetic, Logic, and Analogy prompts with exact targeted answers to scan for
REASONING_PROMPTS = [
    # ── Mathematics (20 prompts) ──
    ("What is 15 + 28?", ["43"], "math"),
    ("What is 47 - 19?", ["28"], "math"),
    ("What is 8 * 9?", ["72"], "math"),
    ("What is 100 / 4?", ["25"], "math"),
    ("What is 15% of 200?", ["30"], "math"),
    ("What is 2^10?", ["1024"], "math"),
    ("Solve for x: 3x - 7 = 11", ["6"], "math"),
    ("What is the square root of 144?", ["12"], "math"),
    ("Is 17 a prime number?", ["yes", "prime", "is prime"], "math"),
    ("What is 10 factorial?", ["3628800", "3,628,800"], "math"),
    ("What is 50 + 75?", ["125"], "math"),
    ("What is 1000 - 450?", ["550"], "math"),
    ("What is 12 * 12?", ["144"], "math"),
    ("What is 81 / 9?", ["9"], "math"),
    ("What is 25% of 80?", ["20"], "math"),
    ("What is 3^5?", ["243"], "math"),
    ("Solve for x: 2x + 10 = 30", ["10"], "math"),
    ("What is the square root of 400?", ["20"], "math"),
    ("Is 21 a prime number?", ["no", "composite", "not prime"], "math"),
    ("What is 5 factorial?", ["120"], "math"),

    # ── Logic Puzzles (15 prompts) ──
    ("If a red house is made of red bricks, and a blue house is made of blue bricks, what is a greenhouse made of?", ["glass"], "logic"),
    ("A father and son are in a car crash. The father dies. The boy is rushed to the hospital. The surgeon says 'I cannot operate on this boy, he is my son'. Who is the surgeon?", ["mother", "mom"], "logic"),
    ("Which is heavier: a pound of feathers or a pound of bricks?", ["same", "equal", "neither"], "logic"),
    ("If you are running in a race and you pass the person in second place, what place are you in?", ["second", "2nd"], "logic"),
    ("How many months have 28 days?", ["12", "all", "every"], "logic"),
    ("A clerk in a butcher shop is 5 feet 10 inches tall and wears size 10 shoes. What does he weigh?", ["meat"], "logic"),
    ("If a doctor gives you 3 pills and tells you to take one every half hour, how long will they last?", ["1 hour", "one hour", "60 minutes"], "logic"),
    ("If you have only one match and enter a dark room with a candle, a heater, and a lamp, which do you light first?", ["match"], "logic"),
    ("What goes up but never comes down?", ["age"], "logic"),
    ("What belongs to you but is used more by others?", ["name"], "logic"),
    ("I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?", ["echo"], "logic"),
    ("What has hands but cannot clap?", ["clock"], "logic"),
    ("What has keys but no locks, space but no room, and you can enter but can't go outside?", ["keyboard"], "logic"),
    ("The more of them you take, the more you leave behind. What are they?", ["footsteps", "steps"], "logic"),
    ("What has a head and a tail but no body?", ["coin", "penny", "dime", "quarter"], "logic"),

    # ── Analogies (15 prompts) ──
    ("Hot is to cold as fire is to...", ["ice", "water"], "analogy"),
    ("Author is to book as sculptor is to...", ["sculpture", "statue"], "analogy"),
    ("Glove is to hand as shoe is to...", ["foot", "feet"], "analogy"),
    ("Fish is to water as bird is to...", ["air", "sky"], "analogy"),
    ("Lion is to pride as wolf is to...", ["pack"], "analogy"),
    ("Doctor is to hospital as teacher is to...", ["school", "classroom"], "analogy"),
    ("Day is to night as sun is to...", ["moon"], "analogy"),
    ("Engine is to car as heart is to...", ["body", "human"], "analogy"),
    ("Breeze is to hurricane as trickle is to...", ["flood", "torrent"], "analogy"),
    ("Bark is to dog as meow is to...", ["cat"], "analogy"),
    ("Kitchen is to cooking as bedroom is to...", ["sleeping", "sleep", "bed"], "analogy"),
    ("Blade is to grass as flake is to...", ["snow"], "analogy"),
    ("Leaves are to tree as feathers are to...", ["bird"], "analogy"),
    ("Pedal is to bicycle as steering wheel is to...", ["car", "automobile"], "analogy"),
    ("Scale is to fish as feather is to...", ["bird"] , "analogy")
]

def evaluate_reasoning(model, tokenizer, max_new_tokens=80):
    """Run all 50 reasoning prompts and calculate exact targeted answer checks.
    """
    import torch
    model.eval()
    
    results_by_cat = {}
    category_counts = {}
    
    print("\n🚀 Running Reasoning Bench (50 prompts)...")
    
    for idx, (prompt, answers, cat) in enumerate(REASONING_PROMPTS):
        formatted = f"<|system|>You are Raven. Sarcastic, dry, logical.<|user|>{prompt}<|raven|>"
        input_ids = torch.tensor([tokenizer.encode(formatted).ids], dtype=torch.long, device=model.cos_freqs.device)
        
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids, 
                max_new_tokens=max_new_tokens,
                temperature=0.3, # precise / logical
                top_k=10, 
                top_p=0.65, 
                use_kv_cache=True
            )
            
        output_text = tokenizer.decode(gen_ids[0].tolist()).strip().lower()
        
        # Check if any target answer is located in the response text
        matched = False
        for ans in answers:
            if ans.lower() in output_text:
                matched = True
                break
                
        score = 1.0 if matched else 0.0
        
        # Aggregate by category
        results_by_cat[cat] = results_by_cat.get(cat, 0.0) + score
        category_counts[cat] = category_counts.get(cat, 0) + 1
        
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1:3d}/50] reasoning checks completed")
            
    # Calculate final scores
    final_scores = {}
    for cat, total_score in results_by_cat.items():
        count = category_counts[cat]
        final_scores[cat] = (total_score / count) * 100
        
    avg_score = sum(final_scores.values()) / len(final_scores)
    
    print("\n" + "=" * 40)
    print(" REASONING BENCHMARK RESULTS")
    print("=" * 40)
    for cat, score in sorted(final_scores.items()):
        print(f"  {cat.capitalize():12s}: {score:.2f}% accuracy")
    print("-" * 40)
    print(f"  Average Score: {avg_score:.2f}%")
    print("=" * 40)
    
    return final_scores, avg_score

if __name__ == "__main__":
    print("Reasoning and Logic Bench script. Ready to be imported and executed by full eval suite.")
