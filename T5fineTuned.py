import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from transformers import RobertaTokenizer


# Load and prepare your dataset
def dataset_preparation(df):
    df = df.rename(columns={"func_documentation_string": "input_text", "func_code_string": "target_text"})
    df = df.drop(columns=["func_code_tokens", "func_documentation_tokens"])
    return df

dataset = pd.read_csv('./Dataset/updatedTrainingDataWithTokens50k.csv')
processed_df = dataset_preparation(dataset)
# Split the dataset into training and validation sets
train_dataset = Dataset.from_pandas(processed_df)
train_dataset, eval_dataset = train_dataset.train_test_split(test_size=0.1).values()


# Tokenization
tokenizer = T5Tokenizer.from_pretrained("t5-small")
# def tokenize_function(examples):
#     return tokenizer(examples["input_text"], examples["target_text"], padding="max_length", truncation=True, max_length=512)

def tokenize_function(examples):
    inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=256)
    targets = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=256)

    # T5 model uses "labels" for target texts
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_datasets = DatasetDict({
    "train": train_dataset.map(tokenize_function, batched=True),
    "validation": eval_dataset.map(tokenize_function, batched=True)
})


# Model
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.eval()



# Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/Models/T5/results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

# Train
trainer.train()
# Evaluate on the validation dataset
eval_metrics = trainer.evaluate()
# Print or access the specific metric you're interested in
print(eval_metrics)


# Save the fine-tuned model
output_model_dir = "./train_models/T5fine_tuned_model"
trainer.save_model(output_model_dir)
output_tokenizer_dir = "./train_models/T5fine_tuned_tokenizer"
tokenizer.save_pretrained(output_tokenizer_dir)

# ... (previous code remains unchanged)

# Load the trained model and tokenizer for inference
model_for_inference = T5ForConditionalGeneration.from_pretrained("./train_models/T5fine_tuned_model")  # Replace with the actual path to your trained model
tokenizer_for_inference = T5Tokenizer.from_pretrained("./train_models/T5fine_tuned_tokenizer")  # Replace with the actual path to your trained tokenizer

# Example input text for inference
generated_outputs = []
for input_seq in test_inputs:

  input_text_for_inference = input_seq

  # Tokenize the input text
  input_ids_for_inference = tokenizer_for_inference.encode(input_text_for_inference, return_tensors="pt")

  # Generate output for inference
  output_ids_for_inference = model_for_inference.generate(input_ids_for_inference, max_length=150, num_beams=6, length_penalty=2.0, early_stopping=True)
  output_text_for_inference = tokenizer_for_inference.decode(output_ids_for_inference[0], skip_special_tokens=True)

  # Print the generated output for inference
  print("Input Text for Inference:", input_text_for_inference)
  print("Generated Output for Inference:", output_text_for_inference)
  generated_outputs.append(output_text_for_inference)