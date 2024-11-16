import argparse
import os
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType

def convert_and_quantize_to_onnx(onnx_model_path, save_dir, max_length=128):
    os.makedirs(save_dir, exist_ok=True)

    # Применение квантования
    quantized_model_path = os.path.join(save_dir, "model_quantized.onnx")
    
    quantize_dynamic(
        onnx_model_path,
        quantized_model_path,
        weight_type=QuantType.QInt8
    )

    print(f"Модель квантована и сохранена: {quantized_model_path}")

    return quantized_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and quantize a Hugging Face model to ONNX format.")
    parser.add_argument('--onnx_model_path', type=str, required=True, help="Path to the trained Hugging Face model.")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the ONNX model.")
    parser.add_argument('--max_length', type=int, default=128, help="Maximum sequence length for tokenization.")
    args = parser.parse_args()

    convert_and_quantize_to_onnx(args.onnx_model_path, args.save_dir, args.max_length)

"""
!python3 ../src/models_dev/quantize.py --onnx_model_path ../models/onnx_model/model.onnx \
                                --save_dir ../models/onnx_model \
                                --max_length 128
"""
