from yoloe_package import load_config, setup_weights_dir, load_parameters, create_directories, run_inference_pipeline

def main():
    """Главная функция для запуска bulk инференса YOLOE.

    Загружает конфигурацию, настраивает параметры,
    создает директории и запускает пайплайн инференса.
    """
    config = load_config()
    setup_weights_dir(config)
    params = load_parameters(config)
    create_directories(params["output_dir"], params["runs_dir"])
    output_dir = run_inference_pipeline(params)
    print(f"Bulk inference completed. Results in {output_dir}")


if __name__ == "__main__":
    main()
