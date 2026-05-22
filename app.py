# app.py
# Premium, dark monochrome Gradio Web UI for RAVEN-2 (HF Spaces compatible)
# Built with a customized developer-first JetBrains Mono interface and performance statistics tracking.

import os
import sys
import time
import torch
import gradio as gr
from tokenizers import Tokenizer

import config
from model import Raven2
from inference import load_model, clean_response, SAMPLING_MODES, WITTY_FALLBACKS

# CSS for a premium dark monochrome developer-centric layout
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');

body {
    background-color: #030303 !important;
}
.gradio-container {
    font-family: 'JetBrains Mono', monospace !important;
    max-width: 800px !important;
    margin: 40px auto !important;
    background-color: #0a0a0a !important;
    color: #e0e0e0 !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 16px !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.8) !important;
    padding: 32px !important;
}
#title_header {
    text-align: center;
    margin-bottom: 24px;
}
#title_header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
    letter-spacing: -1px;
}
#title_header p {
    color: #666666;
    font-size: 0.95rem;
}
.chatbot .user {
    background-color: #111111 !important;
    border: 1px solid #222222 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
.chatbot .bot {
    background-color: #070707 !important;
    border: 1px solid #141414 !important;
    border-radius: 8px !important;
    color: #e0e0e0 !important;
}
.gr-box {
    background-color: #0f0f0f !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 8px !important;
}
input, textarea, select {
    background-color: #121212 !important;
    border: 1px solid #222222 !important;
    color: #ffffff !important;
    font-family: 'JetBrains Mono', monospace !important;
}
input:focus, textarea:focus, select:focus {
    border-color: #444444 !important;
}
.gr-button-secondary {
    background-color: #111111 !important;
    border: 1px solid #222222 !important;
    color: #888888 !important;
    font-weight: 500 !important;
}
.gr-button-secondary:hover {
    background-color: #181818 !important;
    border-color: #333333 !important;
    color: #ffffff !important;
}
.gr-button-primary {
    background-color: #ffffff !important;
    border: 1px solid #ffffff !important;
    color: #000000 !important;
    font-weight: 600 !important;
}
.gr-button-primary:hover {
    background-color: #e0e0e0 !important;
    border-color: #e0e0e0 !important;
    box-shadow: 0 0 15px rgba(255, 255, 255, 0.2) !important;
}
footer {
    display: none !important;
}
"""

# Global references to model and tokenizer
model = None
tokenizer = None
checkpoint_type = None

def init_app():
    global model, tokenizer, checkpoint_type
    
    if not os.path.exists(config.tokenizer_path):
        print("❌ Tokenizer not found. Please train tokenizer first.")
        sys.exit(1)
        
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    
    print("Loading RAVEN-2 model...")
    model, checkpoint_type = load_model()
    print("Web server ready!")

def predict(message, history, system_prompt, sampling_mode, temperature):
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "Model not initialized."
        
    # Build prompt incorporating conversation history (limit to last 5 turns)
    # Gradio history is in format: [[user_msg, bot_msg], ...]
    history_context = history[-5:] if history else []
    
    prompt = f"<|system|>{system_prompt}"
    for user_turn, bot_turn in history_context:
        if user_turn and bot_turn:
            prompt += f"<|user|>{user_turn}<|raven|>{bot_turn}<|eos|>"
            
    # Append current user prompt
    prompt += f"<|system|>{system_prompt}<|user|>{message}<|raven|>"
    
    # Tokenize input
    idx = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=config.device)
    
    # Extract parameters based on selected mode
    p = SAMPLING_MODES.get(sampling_mode, SAMPLING_MODES["default"])
    # If custom temp is shifted from default, override it
    active_temp = temperature if temperature is not None else p["temp"]
    
    t0 = time.time()
    # Generate new tokens using KV Cache
    with torch.no_grad():
        gen_ids = model.generate(
            idx, 
            max_new_tokens=120, 
            temperature=active_temp, 
            top_k=p["top_k"], 
            top_p=p["top_p"], 
            repetition_penalty=p["rep_pen"],
            use_kv_cache=True
        )
    latency = time.time() - t0
    
    # Decode and clean response
    response = tokenizer.decode(gen_ids[0].tolist())
    response = clean_response(response)
    
    # Canned fallback if response generated is empty
    if not response:
        import random
        response = random.choice(WITTY_FALLBACKS)
        
    tok_speed = len(gen_ids[0]) / latency if latency > 0 else 0
    stats = f" [Latency: {latency:.2f}s | Speed: {tok_speed:.1f} tok/s]"
    
    # Yield response (streaming can also be simulated or output directly)
    # Standard ChatInterface outputs response
    return response

def build_interface():
    # Build the gorgeous monochrome web portal
    with gr.Blocks(theme=gr.themes.Monochrome(), css=CUSTOM_CSS) as demo:
        
        with gr.Div(elem_id="title_header"):
            gr.Markdown("# 🐦‍⬛ RAVEN-2")
            gr.Markdown(f"### 100M Parameter AI Agent | Checkpoint: **{checkpoint_type.upper()}**")
            
        with gr.Row():
            # Left panel - System Control Panel
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### 🎛️ CONTROLS")
                
                system_box = gr.Textbox(
                    label="System Prompt",
                    value="You are Raven. Sarcastic, dry, direct, honest.",
                    lines=3,
                    interactive=True
                )
                
                mode_dropdown = gr.Dropdown(
                    label="Sampling Mode",
                    choices=list(SAMPLING_MODES.keys()),
                    value="default",
                    interactive=True
                )
                
                temp_slider = gr.Slider(
                    label="Temperature Override",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    interactive=True
                )
                
                # Stats card
                with gr.Box(elem_id="stats_box"):
                    gr.Markdown("### 📊 INFO")
                    gr.Markdown(f"**Model Size:** 95.18M params  \n"
                                f"**Device:** {config.device.upper()}  \n"
                                f"**Dtype:** {config.dtype_str.upper()}  \n"
                                f"**Context:** 1024 tokens")
                    
            # Right panel - Chat interface
            with gr.Column(scale=3):
                chat = gr.ChatInterface(
                    predict,
                    additional_inputs=[system_box, mode_dropdown, temp_slider],
                    submit_btn="Ask",
                    retry_btn="Retry",
                    undo_btn="Undo",
                    clear_btn="Clear History"
                )
                
        # Link dropdown to temp slider default changes
        def update_slider(mode):
            p = SAMPLING_MODES.get(mode, SAMPLING_MODES["default"])
            return p["temp"]
            
        mode_dropdown.change(update_slider, inputs=[mode_dropdown], outputs=[temp_slider])
        
    return demo

if __name__ == "__main__":
    init_app()
    demo = build_interface()
    # Share = True generates a public gradio link
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
