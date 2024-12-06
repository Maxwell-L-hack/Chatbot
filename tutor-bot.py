from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.intel.neural_compressor import IncQuantizer

model_name="distil-bert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model  = AutoModelForCausalLM.from_pretrained(model_name)

# Apply dynamic quantization
quantizer = IncQuantizer(model)
quantized_model = quantizer.quantize()
quantized_model.save_pretrained("quantized_model")

# Response generator function
def generate_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = quantized_model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shade[-1]:][0], skip_special_tokens=True)
    return response