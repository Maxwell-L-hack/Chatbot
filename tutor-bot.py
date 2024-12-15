import gradio as gr
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad_token_id to eos_token_id
tokenizer.pad_token_id = tokenizer.eos_token_id

# Response generator function
def generate_response(user_input):
    prompt = f"Write a detailed and engaging story about {user_input}:\n\n"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    output = model.generate(
        input_ids, 
        attention_mask=attention_mask,
        max_length=300,  # Increased max_length for longer responses
        min_length=100,  # Increased min_length for more substantial responses
        pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
        temperature=0.7,  # Adjust temperature for more creative responses
        top_p=0.9,  # Use nucleus sampling for more diverse outputs
        do_sample=True  # Enable sampling
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Web interface
def chat(user_input):
    return generate_response(user_input)

iface = gr.Interface(fn=chat, inputs="text", outputs="text", title="Chatbot")
iface.launch(share=True)