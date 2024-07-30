from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the fine-tuned model and tokenizer
model_path = "./my_fine_tuned_model"
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

def generate_text(prompt, max_length=500):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=30,
        top_p=0.92,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def print_wrapped(text, width=80):
    """Print text wrapped to a specified width."""
    import textwrap
    print("\nModel response:")
    print("-" * width)
    for line in textwrap.wrap(text, width=width):
        print(line)
    print("-" * width + "\n")

# Interactive mode
print("Enter your prompts. Type 'quit' to exit.")
while True:
    user_prompt = input("You: ").strip()
    if user_prompt.lower() == 'quit':
        break
    if not user_prompt:
        print("Please enter a non-empty prompt.")
        continue
    generated_text = generate_text(user_prompt)
    print_wrapped(generated_text)