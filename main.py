import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
device = "cpu"
print(f"Model selected: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

SYSTEM_PROMPT = """Your name is Sydney, You are a AI model made by Andrew. You are a helpful assistant.
"""


def run_inference(message: str, history: list):
    history.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})
    chat_message = {'role': 'user', 'content': message}
    history.append(chat_message)
    input_text = tokenizer.apply_chat_template(history, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=256, temperature=0.2, top_p=0.9, do_sample=True)
    decoded = tokenizer.decode(outputs[0])
    response = decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]

    print('=' * 100)
    for h in history:
        print(h)

    return response


if __name__ == '__main__':
    gr.ChatInterface(fn=run_inference, type="messages").launch()
