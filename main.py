import gradio as gr
import ollama

model_list = ollama.list()
if model_list.models:
    print(f"Available models:")
    for m in model_list.models:
        print(f'- {m.model}')
else:
    print('pulling model...')
    ollama.pull('llama3.2:1b')
    model_list = ollama.list()
model_name = model_list.models[0].model
print(f"\nModel selected: {model_name}")

SYSTEM_PROMPT = """Your name is Sydney, you are a AI model made by Andrew. You are a helpful assistant.
"""


def run_inference(message: str, history: list):
    history.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})
    chat_message = {'role': 'user', 'content': message}
    history.append(chat_message)
    stream = ollama.chat(model=model_name, messages=history, stream=True)
    partial_message = ""
    for chunk in stream:
        if len(chunk['message']['content']) != 0:
            partial_message = partial_message + chunk['message']['content']
            yield partial_message

    print('=' * 50)
    for h in history:
        print(h)


if __name__ == '__main__':
    gr.ChatInterface(fn=run_inference, type="messages").launch()
