import json
import torch
from datasets import Dataset, load_from_disk
from transformers import (
    TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForLanguageModeling, AdamW
)

# Define paths
DATA_PATH = "../data/intents.json"
TOKENIZED_PATH = "../data/tokenized_phi2_data"
MODEL_SAVE_PATH = "../models/phi2_finetuned_model"

# Load Phi-2 model and tokenizer
MODEL_NAME = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ Ensure the tokenizer has a pad token (fix padding issue)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True
)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert dataset into training format (User input → AI response)
train_data = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        for response in intent["responses"]:
            train_data.append({"text": f"User: {pattern}\nAI: {response}"})

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

# Create dataset and tokenize it
dataset = Dataset.from_list(train_data)
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Save tokenized dataset
tokenized_dataset.save_to_disk(TOKENIZED_PATH)

# Load tokenized dataset
tokenized_dataset = load_from_disk(TOKENIZED_PATH)

# Split into train & validation sets
train_test = tokenized_dataset.train_test_split(test_size=0.1)
train_data = train_test["train"]
val_data = train_test["test"]

training_args = TrainingArguments(
    output_dir="../models/phi2_finetuned",
    evaluation_strategy="epoch",
    save_strategy="no", 
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=4,
    learning_rate=3e-5,
    weight_decay=0.05,
    fp16=False,
    bf16=False,
    gradient_checkpointing=True,
    max_grad_norm=0.5,
    logging_dir="../logs",
    logging_steps=10,
    push_to_hub=False
)



data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  
)


optimizer = AdamW(model.parameters(), lr=3e-5)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, None)  
)

# Start training
trainer.train()

# Save fine-tuned model
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print(f"✅ Training complete! Fine-tuned model saved at {MODEL_SAVE_PATH}")
