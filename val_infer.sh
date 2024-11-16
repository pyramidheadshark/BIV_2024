python3 src/models_dev/q_infer_val.py \
    --val_data_path data/processed/val_data.tsv \
    --model_path models/onnx_model/model_quantized.onnx \
    --tokenizer_path models/onnx_model \
    --batch_size 64 \
    --max_length 128