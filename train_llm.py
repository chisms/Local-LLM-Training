import json
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def print_sample_data(dataset, num_samples=5):
    print("Sample training data:")
    for i in range(min(num_samples, len(dataset))):
        print(f"Sample {i + 1}:")
        print(dataset[i]['text'][:200] + "...\n")

def prepare_dataset(texts):
    # Ensure texts is a list of strings
    if isinstance(texts, list) and all(isinstance(item, dict) and "text" in item for item in texts):
        texts = [item["text"] for item in texts]
    elif not isinstance(texts, list) or not all(isinstance(item, str) for item in texts):
        raise ValueError("Input must be a list of strings or a list of dictionaries with 'text' key")
    
    dataset = [{"text": chunk} for chunk in texts]
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)
    return Dataset.from_list(train_data), Dataset.from_list(val_data)

def train_model(texts, device):
    print(f"Train_llm.py: Using device: {device}")
    # Prepare the dataset
    train_dataset, val_dataset = prepare_dataset(texts)

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)

    # Tokenize datasets
    def tokenize_function(examples):
        outputs = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
        return {k: torch.tensor(v).to(device) for k, v in outputs.items()}
        # return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=20,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        load_best_model_at_end=True,
        fp16=True,
        gradient_accumulation_steps=4,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    print_sample_data(train_dataset)

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("./my_fine_tuned_model")
    tokenizer.save_pretrained("./my_fine_tuned_model")

    print("Training completed and model saved.")

if __name__ == "__main__":
    # This part is for running the script directly, not through Flask
    file_path = "video_id_train.json"  # Replace with your actual file path
    with open(file_path, 'r') as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    train_model(texts)