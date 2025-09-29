# ======================================
# Step 0: Setup
# ======================================

# Install Hugging Face libraries (only needed in Colab, not in Cursor/local if already installed)
!pip install -q transformers datasets evaluate accelerate

# Check GPU availability
!nvidia-smi
# ======================================
# Step 1: Imports
# ======================================
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np
# ======================================
# Step 2: Load Dataset
# ======================================

# Example: If you have your own CSV file in GitHub, replace with raw link
# dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

# For demo, let's load Sentiment140 (1.6M tweets)
dataset = load_dataset("sentiment140")

# Use a smaller subset first (to test the pipeline)
# dataset["train"] = dataset["train"].shuffle(seed=42).select(range(20000))
# dataset["test"]  = dataset["test"].shuffle(seed=42).select(range(5000))

dataset
# ======================================
# Step 3: Tokenizer
# ======================================
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_fn, batched=True)

# Keep only the needed columns
tokenized_datasets = tokenized_datasets.remove_columns(["text", "date", "query", "user"])
tokenized_datasets = tokenized_datasets.rename_column("sentiment", "labels")
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]
test_dataset  = tokenized_datasets["test"]
# ======================================
# Step 4: Model
# ======================================
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=3
)
# ======================================
# Step 5: Metrics
# ======================================
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    }
# ======================================
# Step 6: TrainingArguments
# ======================================
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro"
)
# ======================================
# Step 7: Trainer
# ======================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
# ======================================
# Step 8: Train
# ======================================
trainer.train()
# ======================================
# Step 9: Evaluate + Save
# ======================================
trainer.evaluate()

# Save model to Colab storage
trainer.save_model("./sentiment-roberta")

# If you want to push to Hugging Face Hub (optional)
# !huggingface-cli login
# trainer.push_to_hub("your-username/sentiment-roberta")
# ======================================
# Step 10: Inference
# ======================================
from transformers import pipeline

pipe = pipeline("text-classification", model="./sentiment-roberta", tokenizer="roberta-base")

print(pipe("I absolutely love this movie!"))
print(pipe("This is the worst day ever."))
print(pipe("It’s just okay, nothing special."))