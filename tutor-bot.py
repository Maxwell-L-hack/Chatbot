from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.intel.neural_compressor import IncQuantizer

model_name="distil-bert-uncased"