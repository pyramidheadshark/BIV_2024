import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

def convert_to_onnx(model_path, onnx_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    print("Конвертация модели в формат ONNX...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_path, from_transformers=True)
    ort_model.save_pretrained(onnx_path)
    print(f"Модель успешно сохранена в формате ONNX: {onnx_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert a model to ONNX format.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument('--onnx_path', type=str, default="./onnx_model", help="Path to save the ONNX model.")
    args = parser.parse_args()

    convert_to_onnx(args.model_path, args.onnx_path)

if __name__ == "__main__":
    main()

"""
python convert_to_onnx.py --model_path <path_to_your_model> --onnx_path <path_to_save_onnx_model>

"""