import argparse
import sys

from model_manager import ModelManager


def main():
    parser = argparse.ArgumentParser(
        description="Download models for Mini Model Server"
    )
    parser.add_argument("--model", type=str, help="Model ID to download")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--all", action="store_true", help="Download all models")

    args = parser.parse_args()
    manager = ModelManager()

    if args.list:
        print("Available models:")
        for mid, data in manager.list_available_models().items():
            downloaded = "✅" if manager.is_model_downloaded(mid) else "❌"
            print(f" - {mid} [{downloaded}] ({data['description']})")
        return

    if args.model:
        print(f"Downloading {args.model}...")
        try:
            manager.download_model(args.model)
            print("Done!")
        except Exception as e:
            print(f"Error: {e}")
        return

    if args.all:
        for mid in manager.list_available_models():
            print(f"Downloading {mid}...")
            try:
                manager.download_model(mid)
            except Exception as e:
                print(f"Error downloading {mid}: {e}")
        print("All downloads processed.")
        return

    # Default behavior if no args: list
    print("Use --list, --model <id>, or --all. Defaults to --list.")
    print("Available models:")
    for mid, data in manager.list_available_models().items():
        downloaded = "✅" if manager.is_model_downloaded(mid) else "❌"
        print(f" - {mid} [{downloaded}] ({data['description']})")


if __name__ == "__main__":
    main()
