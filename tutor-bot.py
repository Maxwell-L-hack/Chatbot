import gradio as gr
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from optimum.intel.neural_compressor import INCModelForMaskedLM
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin", force_download=True)

model_name="distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Apply dynamic quantization
INCModelForMaskedLM.from_pretrained(model_name, quantize=True)
quantized_model.save_pretrained("quantized_model")

# Response generator function
def generate_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = quantized_model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

# Web interface
def chat(user_input):
    return generate_response(user_input)

iface = gr.Interface(fn=chat, inputs="text", outputs="text", title="Educational Tutor Bot")
iface.launch()

dataset = load_dataset("squad")