import numpy as np
from data import load_with_datasets as load_data
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from evaluate import load as load_metric

# define training hyperparameters
TRAINING_ARGUMENTS = [
  TrainingArguments(
    output_dir="./bert/results/1-16",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=25,
    save_steps=35000,
    evaluation_strategy='epoch'
  ),
  TrainingArguments(
    output_dir="./bert/results/4-16",
    learning_rate=4e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=25,
    save_steps=35000,
    evaluation_strategy='epoch'
  ),
  TrainingArguments(
    output_dir="./bert/results/4-32",
    learning_rate=4e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=25,
    save_steps=35000,
    evaluation_strategy='epoch'
  )
]

# training function
def start_train():
  # load and tokenize data
  tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased", padding="max_length", truncation=True, model_max_length=512)
  tokenized_data = load_data().map(lambda x: {
    "text": x["text"],
    "label": 1 if x["label"] == 'yes' else 0
  }).map(lambda x: tokenizer(x['text'], truncation=True, max_length=512))

  # create data collator
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  # load pretrained model
  model = AutoModelForSequenceClassification.from_pretrained("indolem/indobert-base-uncased", num_labels=2)

  # load accuracy metric function
  metric = load_metric("accuracy")
  def compute_metrics(eval_pred):
      logits, labels = eval_pred
      predictions = np.argmax(logits, axis=-1)
      return metric.compute(predictions=predictions, references=labels)

  # start training for each hyperparameter
  for training_args in TRAINING_ARGUMENTS:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()