import argparse
import onnx
import onnxoptimizer  # You may need to install onnxoptimizer (pip install onnxoptimizer)

def optimize_onnx(onnx_path, optimized_onnx_path, optimization_level):
    # Load the ONNX model
    print("Загрузка модели ONNX...")
    model = onnx.load(onnx_path)

    # Determine the optimization passes based on optimization level
    if optimization_level == 0:
        print("Оптимизация уровня 0: Нет оптимизаций.")
        optimized_model = model
    elif optimization_level == 1:
        print("Оптимизация уровня 1: Основные оптимизации.")
        # Example: Apply a subset of optimization passes
        optimized_model = onnxoptimizer.optimize(model, passes=["eliminate_deadend", "fuse_consecutive_transposes"])
    elif optimization_level == 2:
        print("Оптимизация уровня 2: Полные оптимизации.")
        # Example: Apply more aggressive optimizations
        optimized_model = onnxoptimizer.optimize(model)
    else:
        raise ValueError("Неверный уровень оптимизации. Используйте 0, 1 или 2.")

    # Save the optimized ONNX model
    onnx.save(optimized_model, optimized_onnx_path)
    print(f"Оптимизированная ONNX модель успешно сохранена: {optimized_onnx_path}")

def main():
    parser = argparse.ArgumentParser(description="Optimize an ONNX model.")
    parser.add_argument('--onnx_path', type=str, required=True, help="Path to the ONNX model file.")
    parser.add_argument('--optimized_onnx_path', type=str, default="./onnx_optimized_model.onnx", help="Path to save the optimized ONNX model.")
    parser.add_argument('--optimization_level', type=int, default=2, help="Optimization level (0, 1, or 2).")
    args = parser.parse_args()

    optimize_onnx(args.onnx_path, args.optimized_onnx_path, args.optimization_level)

if __name__ == "__main__":
    main()

"""
Usage example:
python optimize_onnx.py --onnx_path <path_to_your_onnx_model> --optimized_onnx_path <path_to_save_optimized_model> --optimization_level 2
"""
