from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.intel.neural_compressor import IncQuantizer

model_name="distil-bert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model  = AutoModelForCausalLM.from_pretrained(model_name)

# Apply dynamic quantization
quantizer = IncQuantizer(model)
quantized_model = quantizer.quantize()
quantized_model.save_pretrained("quantized_model")

