from yoloe_package import load_config, setup_weights_dir, load_parameters, create_directories, run_inference_pipeline

def main():
    """Главная функция для запуска bulk инференса YOLOE.

    Загружает конфигурацию, настраивает параметры,
    создает директории и запускает пайплайн инференса.
    """
    print("🚀 === Starting YOLOE Bulk Inference Pipeline ===")

    print("1. 🔧 Loading configuration...")
    config = load_config()

    print("2. 📁 Setting up weights directory...")
    setup_weights_dir(config)

    print("3. ⚙️  Loading parameters...")
    params = load_parameters(config)

    # Check SAHI status
    if params.get("use_sahi", False):
        try:
            import sahi
            print("4. 🎯 SAHI tiled inference: ENABLED")
            print(f"   Slice size: {params['sahi_slice_height']}x{params['sahi_slice_width']}")
            print(f"   Overlap: {params['sahi_overlap_height_ratio']} (H) x {params['sahi_overlap_width_ratio']} (W)")
        except ImportError:
            print("4. ⚠️  SAHI tiled inference: REQUESTED but SAHI not installed!")
            print("   Falling back to standard inference...")
            params["use_sahi"] = False
    else:
        print("4. 🔍 Standard inference mode")

    print("5. 📂 Creating directories...")
    create_directories(params["output_dir"], params["runs_dir"])

    print("6. 🚀 Running inference pipeline...")
    output_dir = run_inference_pipeline(params)

    print("7. 💾 Exporting results...")
    print("🎉 === Bulk inference completed successfully! ===")
    print(f"📁 Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
