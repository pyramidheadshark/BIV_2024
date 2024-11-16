import argparse
import pandas as pd
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def preprocess_data(df):
    df['text'] = df.apply(lambda x: f"{x['date']} [SEP] {x['amount']} [SEP] {x['description']}", axis=1)
    return df['text'], df['category']

def tokenize_data(batch, tokenizer, max_length):
    encoding = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="np")
    return encoding

def load_onnx_model(onnx_model_path):
    return ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

def infer_batch(session, inputs):
    ort_inputs = {k: v for k, v in inputs.items()}
    ort_outputs = session.run(None, ort_inputs)
    return np.argmax(ort_outputs[0], axis=1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a quantized ONNX model on validation data.")
    parser.add_argument('--val_data_path', type=str, required=True, help="Path to the validation data file.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the quantized ONNX model.")
    parser.add_argument('--tokenizer_path', type=str, required=True, help="Path to the tokenizer directory.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for inference.")
    parser.add_argument('--max_length', type=int, default=128, help="Maximum sequence length for tokenization.")
    args = parser.parse_args()

    val_data = pd.read_csv(args.val_data_path, sep='\t')
    val_texts, val_labels = preprocess_data(val_data)

    val_labels = val_labels.astype('category').cat.codes

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    session = load_onnx_model(args.model_path)

    predictions = []
    for i in tqdm(range(0, len(val_texts), args.batch_size), desc="Inferencing"):
        batch_texts = val_texts[i:i + args.batch_size].tolist()
        tokenized_batch = tokenize_data(batch_texts, tokenizer, args.max_length)
        batch_predictions = infer_batch(session, tokenized_batch)
        predictions.extend(batch_predictions)

    print(f"True labels: {val_labels[:10].tolist()}")
    print(f"Predictions: {predictions[:10]}")

    acc = accuracy_score(val_labels, predictions)
    f1 = f1_score(val_labels, predictions, average='weighted')

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
