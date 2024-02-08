#By: VinsmokeSomya from DumbKid

# Install the required libraries
!pip install transformers accelerate
!pip install googletrans==4.0.0-rc1

# Import necessary libraries
from transformers import AutoTokenizer, pipeline
import transformers
import torch
from googletrans import Translator

# Set up the model and tokenizer
model = "DumbKid-AI007/Espada-S1-7B-mr"
tokenizer = AutoTokenizer.from_pretrained(model)
generator = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Get user input for the initial prompt
initial_prompt = input("Enter the initial prompt: ")

# Generate text
sequences = generator(
    f'You: {initial_prompt} Espada:',
    max_length=1000,
    num_return_sequences=1,
)

# Print Generated text
for seq in sequences:
    generated_text = seq['generated_text']
    print(f"{generated_text}")