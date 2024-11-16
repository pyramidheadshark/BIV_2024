python src\models_dev\q_infer.py ^
    --file_path payments_main.tsv ^
    --model_path models\onnx_model\model_quantized.onnx ^
    --tokenizer_path models\onnx_model ^
    --output_file data\predictions.tsv ^
    --batch_size 64 ^
    --max_length 128