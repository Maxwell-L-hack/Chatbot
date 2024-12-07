import gradio as gr
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from optimum.intel.neural_compressor import INCModelForMaskedLM

model_name="distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("distilbert/distilbert-base-uncased")

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