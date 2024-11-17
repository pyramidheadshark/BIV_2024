import argparse
import pandas as pd
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
from tqdm import tqdm

def preprocess_data(df):
    df['text'] = df.apply(lambda x: f"{x['description']}", axis=1)
    return df['text']

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
    parser = argparse.ArgumentParser(description="Inference using a quantized ONNX model.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the inference data file.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the quantized ONNX model file.")
    parser.add_argument('--tokenizer_path', type=str, required=True, help="Path to the tokenizer directory.")
    parser.add_argument('--output_file', type=str, default="predictions.csv", help="File to save predictions.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for inference.")
    parser.add_argument('--max_length', type=int, default=128, help="Maximum sequence length for tokenization.")
    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.file_path, sep='\t', header=None, names=['id', 'date', 'amount', 'description', 'category'])
    texts = preprocess_data(data)

    # Load tokenizer and ONNX model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    session = load_onnx_model(args.model_path)

    index_to_category = {
        0: 'BANK_SERVICE',
        1: 'FOOD_GOODS',
        2: 'LEASING',
        3: 'LOAN',
        4: 'NON_FOOD_GOODS',
        5: 'NOT_CLASSIFIED',
        6: 'REALE_STATE',
        7: 'SERVICE',
        8: 'TAX'
    }

    # Tokenize and batch data
    predictions = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Inferencing"):
        batch_texts = texts[i:i + args.batch_size].tolist()
        tokenized_batch = tokenize_data(batch_texts, tokenizer, args.max_length)
        batch_predictions = infer_batch(session, tokenized_batch)
        predictions.extend(batch_predictions)

    data['predicted_category'] = [index_to_category[idx] for idx in predictions]

    data[['id', 'predicted_category']].to_csv(args.output_file, header=None, sep='\t',  index=False)
    print(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()


