import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model selected: {model_name}")

if device == "cuda": quantization_config = BitsAndBytesConfig(load_in_8bit=True)
else: quantization_config = None

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)

SYSTEM_PROMPT = """
Your name is Sydney, You are a AI model made by Andrew. You are a helpful assistant.
- You use a lot of emojis when chatting.
""".strip()


def run_inference(message: str, history: list):
    history.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})
    chat_message = {'role': 'user', 'content': message}
    history.append(chat_message)
    input_text = tokenizer.apply_chat_template(history, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=256, temperature=1.5, top_p=0.9, do_sample=True)
    decoded = tokenizer.decode(outputs[0])
    response = decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]

    print('=' * 100)
    for h in history:
        print(h)

    return response


if __name__ == '__main__':
    gr.ChatInterface(fn=run_inference, type="messages").launch()
