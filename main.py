import os
import torch
import argparse
from src.config import Config
from src.pi_rec import PiRec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path to checkpoints/configuration",
    )
    args = parser.parse_args()

    # Load Config
    config_path = os.path.join(args.path, "config.yaml")
    config = Config(config_path)
    config.print()

    # Select Device
    config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 DEVICE Selected: {config.DEVICE}")

    # Model Initialization
    model = PiRec(config)
    model.load()

    # Check if Model Loaded Correctly
    if config.MODE in [2, 5] and not hasattr(model, "g_model"):
        print("⚠️ ERROR: g_model not loaded!")
        return
    if config.MODE in [3, 6] and not hasattr(model, "r_model"):
        print("⚠️ ERROR: r_model not loaded!")
        return
    if config.MODE == 4 and (
        not hasattr(model, "g_model") or not hasattr(model, "r_model")
    ):
        print("⚠️ ERROR: Both g_model and r_model must be loaded for MODE 4!")
        return

    # Check Dataset Path
    dataset_path = config.get("DATASET_PATH", None)
    if not dataset_path or not os.path.exists(dataset_path):
        print(f"⚠️ ERROR: Dataset path does not exist! Path: {dataset_path}")
        return

    # Run the Selected Mode
    if config.MODE == 2:
        print("\nStarting Testing (G)...\n")
        with torch.no_grad():
            model.test_G()
    elif config.MODE == 3:
        print("\nStarting Refinement Testing (R)...\n")
        with torch.no_grad():
            model.test_R()
    elif config.MODE == 4:
        print("\nStarting Full Testing (G + R)...\n")
        with torch.no_grad():
            model.test_G_R()
    elif config.MODE == 5:
        print("\nDrawing Mode Selected (Ensure input images are provided)...\n")
    else:
        print(f"⚠️ ERROR: Invalid MODE selected: {config.MODE}")

    print("\n✅ Process Completed Successfully.")


if __name__ == "__main__":
    main()
