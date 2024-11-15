import argparse
import pandas as pd
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import torch
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара

def preprocess_data(df):
    df['text'] = df.apply(lambda x: f"{x['date']} [SEP] {x['amount']} [SEP] {x['description']}", axis=1)
    return df[['id', 'text']]

def predict_batch(texts, model, tokenizer, batch_size=32):
    results = []
    # Используем tqdm для отслеживания прогресса
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches", unit="batch"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, axis=1).cpu().numpy()
            results.extend(predictions)
    return results

def infer_from_tsv(input_path, model, tokenizer, output_path, batch_size):
    data = pd.read_csv(input_path, sep='\t', header=None, names=['id', 'date', 'amount', 'description'])
    
    data = preprocess_data(data)
    
    predictions = predict_batch(data['text'].tolist(), model, tokenizer, batch_size=batch_size)
    
    data['predicted_category'] = predictions
    data.to_csv(output_path, sep='\t', index=False)
    print(f"Предсказания сохранены в {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Inference for text classification using an optimized ONNX model.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input TSV file.")
    parser.add_argument('--onnx_model_path', type=str, required=True, help="Path to the optimized ONNX model.")
    parser.add_argument('--output_path', type=str, default="./predictions.csv", help="Path to save the predictions.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for inference.")
    args = parser.parse_args()

    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(args.onnx_model_path)
    model = ORTModelForSequenceClassification.from_pretrained(args.onnx_model_path)

    infer_from_tsv(args.input_path, model, tokenizer, args.output_path, args.batch_size)

if __name__ == "__main__":
    main()

"""
!python3 infer.py --input_path data/raw/payments_main.tsv \
                 --onnx_model_path models/onnx_optimized_model \
                 --output_path ./predictions.csv \
                 --batch_size 64
"""
