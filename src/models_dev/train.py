import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def preprocess_data(df):
    df['text'] = df.apply(lambda x: f"{x['description']}", axis=1)
    df = df[['text', 'category']]
    
    df['category'] = df['category'].astype('category').cat.codes
    return df

def tokenize_data(batch, tokenizer):
    encoding = tokenizer(batch['text'], padding=True, truncation=True, max_length=128)
    encoding['labels'] = batch['category']
    return encoding

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

def main():
    parser = argparse.ArgumentParser(description="Train a text classification model.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the training data file.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to model.")
    parser.add_argument('--model_name', type=str, default="ai-forever/ru-en-RoSBERTa", help="Pretrained model name.")
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save the results.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate.")
    parser.add_argument('--max_length', type=int, default=128, help="Maximum sequence length for tokenization.")
    args = parser.parse_args()

    # Загрузка данных
    data = pd.read_csv(args.file_path, sep='\t', header=None, names=['date', 'amount', 'description', 'category'])
    train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['category'], random_state=42)

    train_data = preprocess_data(train_data)
    val_data = preprocess_data(val_data)

    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(data['category'].unique()))

    train_dataset = train_dataset.map(lambda batch: tokenize_data(batch, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda batch: tokenize_data(batch, tokenizer), batched=True)

    train_dataset = train_dataset.remove_columns(['text', '__index_level_0__'])
    val_dataset = val_dataset.remove_columns(['text', '__index_level_0__'])

    # Приведение данных в формат PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        save_strategy='steps',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        evaluation_strategy="steps",
        eval_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    main()

