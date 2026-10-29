import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def pick_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device

def load_model(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    device = pick_device()
    print(f"Loading model {model_name} on device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    print(f"Loaded model {model_name} on device {device}")
    return model, tokenizer

def generate_text(model, tokenizer, user_text, system_prompt=None, max_new_tokens=120, **decoding):
    """ Send one message to the model and get the response. """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, **decoding)
    new_tokens = outputs[0][inputs.input_ids.shape[1]:].len()
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return generated_text

import pandas as pd
# from src.minutes.engine import load_model, generate_text

model, tokenizer = load_model("Qwen/Qwen2.5-1.5B-Instruct")
dev = pd.read_csv("data/dev_samples.csv")
dialogue = dev.iloc[0]["dialogue"]

answer = generate_text(model, tokenizer, user_text=dialogue, system_prompt="You are a helpful assistant that summarizes conversations.", do_sample=False)

print(answer)
print("\n Human summary:\n", dev.iloc[0]["summary"])