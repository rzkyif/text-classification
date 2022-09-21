import numpy as np
import time
from data import load_with_datasets as load_data
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from evaluate import load as load_metric

from .train import TRAINING_ARGUMENTS

def start_test():
  MODELS = [('BASE', None)] + [(x.output_dir[-4:], x) for x in TRAINING_ARGUMENTS]

  for (model_id, model_args) in MODELS:

    print(f"----------------- MODEL {model_id} START -----------------")

    model_path = f'./bert/results/{model_id}' if model_id != 'BASE' else 'indolem/indobert-base-uncased'

    print(f"\nLoading model...")
    s_time = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, padding="max_length", truncation=True, model_max_length=512)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print(f"...loaded! ({time.perf_counter() - s_time:.2f}ms)")

    print(f"\nLoading test data...")
    s_time = time.perf_counter()
    tokenized_data = load_data().map(lambda x: {
      "text": x["text"],
      "label": 1 if x["label"] == 'yes' else 0
    }).map(lambda x: tokenizer(x['text'], truncation=True, max_length=512))
    print(f"...loaded! ({time.perf_counter() - s_time:.2f}ms)")

    print(f"\nPreparing metrics...")
    s_time = time.perf_counter()
    metric = load_metric("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    print(f"...metrics ready! ({time.perf_counter() - s_time:.2f}ms)")

    print(f"\nEvaluating model '{model_id}'...")
    s_time = time.perf_counter()
    trainer = Trainer(
        model=model,
        args=model_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    metrics = trainer.evaluate()
    print(f"...finished! ({time.perf_counter() - s_time:.2f}ms)")

    print(f"\nEvaluation results:\n{metrics}")

    print(f"\n------------------ MODEL {model_id} END ------------------")