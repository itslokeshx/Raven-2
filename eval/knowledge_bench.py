# eval/knowledge_bench.py
# Factual Knowledge Benchmark for RAVEN-2 (100M parameter LLM)
# Contains 100 diverse knowledge prompts across 5 key categories: Science, History, Geography, People, and Concepts.
# Evaluates factual accuracy based on exact-match or key terms overlap ratio.

import os
import sys

# Knowledge Questions and Golden Keywords to check in response
KNOWLEDGE_PROMPTS = [
    # ── Science (20 prompts) ──
    ("What causes thunder?", ["lightning", "expansion", "shockwave", "heat", "air"], "science"),
    ("Define photosynthesis.", ["light", "chemical", "energy", "plants", "carbon", "oxygen"], "science"),
    ("What is DNA?", ["genetic", "molecule", "acid", "instructions", "organism"], "science"),
    ("Explain the speed of light.", ["vacuum", "299", "300", "speed", "km", "second"], "science"),
    ("What is gravity?", ["mass", "pulling", "gravitation", "attraction", "force"], "science"),
    ("What is general relativity?", ["einstein", "geometric", "gravity", "space", "time"], "science"),
    ("What is entropy?", ["disorder", "measure", "thermodynamics", "randomness", "uncertainty"], "science"),
    ("Define quantum mechanics.", ["scale", "atoms", "subatomic", "particles", "physics"], "science"),
    ("What is plate tectonics?", ["movement", "lithosphere", "crust", "plates", "earth"], "science"),
    ("What is cellular respiration?", ["glucose", "energy", "oxygen", "cells", "metabolic"], "science"),
    ("Explain absolute zero.", ["temperature", "kelvin", "motion", "celsius", "minimum"], "science"),
    ("What is an isotope?", ["neutrons", "protons", "atomic", "mass", "element"], "science"),
    ("What is covalent bonding?", ["electrons", "atoms", "sharing", "molecular", "bonds"], "science"),
    ("Explain black holes.", ["gravity", "escape", "light", "singularity", "horizon"], "science"),
    ("What is sound?", ["vibration", "wave", "medium", "pressure", "frequency"], "science"),
    ("Define mitosis.", ["cell", "division", "chromosomes", "reproduction", "daughter"], "science"),
    ("What is a supernova?", ["explosion", "star", "collapse", "massive", "brightness"], "science"),
    ("Explain electromagnetism.", ["charge", "force", "magnetic", "electric", "field"], "science"),
    ("What is the ozone layer?", ["atmosphere", "ultraviolet", "radiation", "oxygen", "shield"], "science"),
    ("Define kinetic energy.", ["motion", "mass", "velocity", "work", "movement"], "science"),
    
    # ── History (20 prompts) ──
    ("When did World War II end?", ["1945", "allies", "japan", "germany", "surrender"], "history"),
    ("Who was Julius Caesar?", ["rome", "general", "dictator", "emperor", "senate", "assassinated"], "history"),
    ("What was the French Revolution?", ["monarchy", "bastille", "france", "1789", "equality"], "history"),
    ("What was the Renaissance?", ["rebirth", "art", "italy", "revival", "culture", "science"], "history"),
    ("Who was Abraham Lincoln?", ["president", "civil", "war", "slavery", "union", "assassinated"], "history"),
    ("What was the Magna Carta?", ["king", "john", "1215", "rights", "charter", "law"], "history"),
    ("What is the history of the Pyramids of Giza?", ["egypt", "tombs", "pharaohs", "khufu", "monument"], "history"),
    ("Who wrote the US Declaration of Independence?", ["jefferson", "thomas", "1776", "congress", "independence"], "history"),
    ("What was the Cold War?", ["soviet", "usa", "communist", "tension", "proxy", "nuclear"], "history"),
    ("Explain the fall of the Roman Empire.", ["barbarian", "division", "economic", "rome", "476"], "history"),
    ("Who was Mahatma Gandhi?", ["india", "independence", "nonviolence", "british", "civil"], "history"),
    ("What was the industrial revolution?", ["machines", "steam", "factories", "manufacturing", "century"], "history"),
    ("Who was Joan of Arc?", ["france", "hundred", "years", "war", "orleans", "stake"], "history"),
    ("Explain the treaty of Versailles.", ["world", "war", "germany", "1919", "reparations"], "history"),
    ("What was the Black Death?", ["plague", "pandemic", "fleas", "europe", "rats"], "history"),
    ("Who was Alexander the Great?", ["macedon", "empire", "conquest", "persia", "general"], "history"),
    ("What was the Silk Road?", ["trade", "china", "europe", "route", "merchant"], "history"),
    ("Who was Genghis Khan?", ["mongol", "empire", "conquest", "nomad", "leader"], "history"),
    ("What was the space race?", ["moon", "apollo", "soviet", "usa", "landing", "sputnik"], "history"),
    ("Explain the American Civil War.", ["slavery", "union", "confederacy", "lincoln", "secession"], "history"),

    # ── Geography (20 prompts) ──
    ("What is the capital of France?", ["paris", "seine", "capital"], "geography"),
    ("What is the longest river in the world?", ["nile", "amazon", "river", "africa"], "geography"),
    ("Where is Mount Everest?", ["nepal", "himalayas", "tibet", "mountain", "highest"], "geography"),
    ("What is the capital of Japan?", ["tokyo", "japan", "capital"], "geography"),
    ("Name the five oceans.", ["pacific", "atlantic", "indian", "arctic", "southern"], "geography"),
    ("What is the capital of Australia?", ["canberra", "australia", "capital"], "geography"),
    ("Where is the Sahara Desert?", ["africa", "desert", "northern", "largest"], "geography"),
    ("What is the capital of Canada?", ["ottawa", "ontario", "capital"], "geography"),
    ("Which country has the largest population?", ["china", "india", "population", "largest"], "geography"),
    ("Where are the Andes mountains?", ["south", "america", "range", "mountain"], "geography"),
    ("What is the capital of India?", ["delhi", "new", "capital"], "geography"),
    ("What is the largest country by land area?", ["russia", "siberia", "area", "largest"], "geography"),
    ("Name the largest lake in the world.", ["caspian", "sea", "lake", "freshwater"], "geography"),
    ("Where is the Great Barrier Reef?", ["australia", "reef", "coral", "ocean"], "geography"),
    ("What is the capital of Brazil?", ["brasilia", "brazil", "capital"], "geography"),
    ("Where is Greenland?", ["denmark", "island", "arctic", "largest"], "geography"),
    ("What is the capital of Egypt?", ["cairo", "nile", "capital"], "geography"),
    ("Which is the smallest continent?", ["australia", "continent", "smallest"], "geography"),
    ("Where is the Amazon Rainforest?", ["south", "america", "brazil", "forest", "amazon"], "geography"),
    ("What is the capital of Germany?", ["berlin", "germany", "capital"], "geography"),

    # ── People (20 prompts) ──
    ("Who was Albert Einstein?", ["physicist", "relativity", "photoelectric", "nobel", "physics"], "people"),
    ("Who wrote Hamlet?", ["shakespeare", "william", "playwright", "tragedy", "english"], "people"),
    ("Who painted the Mona Lisa?", ["da vinci", "leonardo", "italian", "artist", "renaissance"], "people"),
    ("Who was Martin Luther King Jr.?", ["civil", "rights", "dream", "speech", "activist", "nonviolence"], "people"),
    ("Who was Marie Curie?", ["radiation", "nobel", "chemistry", "physics", "polonium", "radium"], "people"),
    ("Who was Steve Jobs?", ["apple", "iphone", "macintosh", "pixar", "co-founder"], "people"),
    ("Who was Isaac Newton?", ["gravity", "calculus", "laws", "motion", "physicist"], "people"),
    ("Who is Elon Musk?", ["tesla", "spacex", "x", "paypal", "billionaire"], "people"),
    ("Who was Vincent van Gogh?", ["artist", "starry", "night", "sunflowers", "painter", "dutch"], "people"),
    ("Who was Nelson Mandela?", ["south", "africa", "apartheid", "president", "prison"], "people"),
    ("Who was Charles Darwin?", ["evolution", "natural", "selection", "origin", "species"], "people"),
    ("Who was Wolfgang Amadeus Mozart?", ["composer", "music", "classical", "austrian", "opera"], "people"),
    ("Who is Bill Gates?", ["microsoft", "windows", "philanthropist", "co-founder", "foundation"], "people"),
    ("Who was Winston Churchill?", ["prime", "minister", "britain", "world", "war"], "people"),
    ("Who was Cleopatra?", ["egypt", "queen", "pharaoh", "caesar", "antony"], "people"),
    ("Who was George Washington?", ["president", "general", "revolutionary", "father"], "people"),
    ("Who was Stephen Hawking?", ["cosmologist", "black", "holes", "radiation", "physics"], "people"),
    ("Who was Nikola Tesla?", ["alternating", "current", "induction", "motor", "electricity"], "people"),
    ("Who was Alan Turing?", ["enigma", "computer", "turing", "machine", "intelligence"], "people"),
    ("Who was Sigmund Freud?", ["psychoanalysis", "unconscious", "dream", "psychologist"], "people"),

    # ── Concepts (20 prompts) ──
    ("What is inflation in economics?", ["prices", "currency", "purchasing", "money", "economic"], "concepts"),
    ("Explain supply and demand.", ["market", "price", "consumers", "sellers", "equilibrium"], "concepts"),
    ("What is capitalism?", ["private", "ownership", "market", "capital", "profit"], "concepts"),
    ("Define socialism.", ["social", "ownership", "means", "production", "equality"], "concepts"),
    ("What is democracy?", ["government", "people", "voting", "elections", "representation"], "concepts"),
    ("Explain the concept of globalization.", ["international", "trade", "interconnected", "global", "integration"], "concepts"),
    ("What is checks and balances in government?", ["branches", "separation", "power", "legislative", "executive", "judicial"], "concepts"),
    ("Define blockchain.", ["distributed", "ledger", "cryptography", "decentralized", "transactions"], "concepts"),
    ("What is cryptography?", ["encryption", "security", "data", "decryption", "keys"], "concepts"),
    ("Explain dark matter.", ["universe", "gravity", "invisible", "mass", "galaxies"], "concepts"),
    ("What is plate tectonics?", ["lithosphere", "plates", "crust", "earthquake", "drifting"], "concepts"),
    ("What is a solar eclipse?", ["moon", "sun", "earth", "shadow", "blocking"], "concepts"),
    ("Explain machine learning.", ["data", "algorithms", "train", "learning", "predict"], "concepts"),
    ("What is standard deviation?", ["statistics", "variance", "mean", "spread", "distribution"], "concepts"),
    ("Explain game theory.", ["decision", "strategies", "players", "rational", "outcomes"], "concepts"),
    ("What is the prisoner's dilemma?", ["cooperate", "defect", "betray", "game", "strategy"], "concepts"),
    ("Explain plate tectonics theory.", ["crust", "lithosphere", "drift", "mantle", "geological"], "concepts"),
    ("What is the placebo effect?", ["treatment", "placebo", "psychological", "patient", "medical"], "concepts"),
    ("Define standard model of physics.", ["particles", "fundamental", "forces", "quarks", "boson"], "concepts"),
    ("Explain the term 'cognitive dissonance'.", ["contradictory", "beliefs", "psychological", "discomfort", "tension"], "concepts")
]

def evaluate_knowledge(model, tokenizer, max_new_tokens=60):
    """Run all 100 knowledge prompts and calculate keyword overlap accuracy.
    """
    import torch
    model.eval()
    
    results_by_cat = {}
    category_counts = {}
    
    print("\n🚀 Running Knowledge Bench (100 prompts)...")
    
    for idx, (prompt, keywords, cat) in enumerate(KNOWLEDGE_PROMPTS):
        formatted = f"<|system|>You are Raven. Fact-based agent.<|user|>{prompt}<|raven|>"
        input_ids = torch.tensor([tokenizer.encode(formatted).ids], dtype=torch.long, device=model.cos_freqs.device)
        
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids, 
                max_new_tokens=max_new_tokens,
                temperature=0.4, # precise
                top_k=15, 
                top_p=0.7, 
                use_kv_cache=True
            )
            
        output_text = tokenizer.decode(gen_ids[0].tolist()).strip().lower()
        
        # Check keyword presence
        matched = 0
        for kw in keywords:
            if kw.lower() in output_text:
                matched += 1
                
        score = matched / len(keywords) if keywords else 0.0
        
        # Aggregate by category
        results_by_cat[cat] = results_by_cat.get(cat, 0.0) + score
        category_counts[cat] = category_counts.get(cat, 0) + 1
        
        if (idx + 1) % 25 == 0:
            print(f"  [{idx+1:3d}/100] factual checks completed")
            
    # Calculate final scores
    final_scores = {}
    for cat, total_score in results_by_cat.items():
        count = category_counts[cat]
        final_scores[cat] = (total_score / count) * 100
        
    avg_score = sum(final_scores.values()) / len(final_scores)
    
    print("\n" + "=" * 40)
    print(" KNOWLEDGE BENCHMARK RESULTS")
    print("=" * 40)
    for cat, score in sorted(final_scores.items()):
        print(f"  {cat.capitalize():12s}: {score:.2f}% accuracy")
    print("-" * 40)
    print(f"  Average Score: {avg_score:.2f}%")
    print("=" * 40)
    
    return final_scores, avg_score

if __name__ == "__main__":
    print("Factual Knowledge Bench script. Ready to be imported and executed by full eval suite.")
