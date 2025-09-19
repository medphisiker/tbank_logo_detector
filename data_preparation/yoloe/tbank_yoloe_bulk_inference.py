from yoloe_package import load_config, setup_weights_dir, load_parameters, create_directories, run_inference_pipeline

def main():
    """Главная функция для запуска bulk инференса YOLOE.

    Загружает конфигурацию, настраивает параметры,
    создает директории и запускает пайплайн инференса.
    """
    print("=== Starting YOLOE Bulk Inference Pipeline ===")

    print("1. Loading configuration...")
    config = load_config()

    print("2. Setting up weights directory...")
    setup_weights_dir(config)

    print("3. Loading parameters...")
    params = load_parameters(config)

    print("4. Creating directories...")
    create_directories(params["output_dir"], params["runs_dir"])

    print("5. Running inference pipeline...")
    output_dir = run_inference_pipeline(params)

    print("6. Exporting results...")
    print(f"=== Bulk inference completed successfully! ===")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
