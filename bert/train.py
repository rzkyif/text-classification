from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from data import load_with_datasets

TRAINING_ARGUMENTS = [
  TrainingArguments(
    output_dir="./bert/results/5-8",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1
  ),
  TrainingArguments(
    output_dir="./bert/results/2-16",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1
  ),
  TrainingArguments(
    output_dir="./bert/results/2-8",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1
  )
]

def start_train():
  tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
  tokenized_data = load_with_datasets().map(lambda x: {
    "text": x["text"],
    "label": 1 if x["label"] == 'yes' else 0
  }).map(lambda x: tokenizer(x['text']))
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
  model = AutoModelForSequenceClassification.from_pretrained("indolem/indobert-base-uncased", num_labels=2)

  for training_args in TRAINING_ARGUMENTS:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()