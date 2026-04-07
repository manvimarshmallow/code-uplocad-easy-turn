import os
import shutil
from huggingface_hub import snapshot_download

# --- Configuration ---
# The Hugging Face repository ID for the dataset.
DATASET_ID = "ASLP-lab/Easy-Turn-Testset"

# Where to save the dataset relative to this script's location.
# This will create a folder named 'datasets/Easy-Turn-Testset' in your project root.
DESTINATION_FOLDER = "datasets"
# --- End of Configuration ---

def download_and_place_dataset():
    """
    Downloads the Easy-Turn test set from Hugging Face and places it
    in a local 'datasets' directory.
    """
    # Get the absolute path of the directory where this script is located (your Easy-Turn repo root).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the final destination path for the dataset.
    final_dest_path = os.path.join(script_dir, DESTINATION_FOLDER, DATASET_ID.split('/')[-1])

    print("=" * 50)
    print(f"Target Destination: {final_dest_path}")
    print("=" * 50)
    
    if os.path.exists(final_dest_path):
        print("Dataset already exists at the target destination. Skipping download.")
        print("To re-download, please delete the folder and run this script again.")
        return

    try:
        # 1. Download the entire dataset repository to the Hugging Face cache.
        #    This function is smart and will resume if interrupted.
        print(f"Downloading '{DATASET_ID}' from Hugging Face...")
        print("This may take a few moments. The data is about 145 MB.")
        
        cache_path = snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            # We can ignore the .gitattributes file to save a little space.
            ignore_patterns=[".gitattributes"],
        )
        print("Download complete. Files are now in the cache.")

        # The actual audio files are inside a 'testset' subfolder in the downloaded repo.
        source_path = os.path.join(cache_path, "testset")
        
        # 2. Copy the files from the cache to our desired local directory.
        print(f"\nCopying files from cache to '{final_dest_path}'...")
        
        # We use shutil.copytree to copy the entire folder structure.
        shutil.copytree(source_path, final_dest_path)
        
        print("\n" + "="*50)
        print("✅ Success!")
        print(f"The Easy-Turn test set has been saved to:")
        print(f"   {final_dest_path}")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your internet connection and try again.")
        print("If the issue persists, you may need to log in to Hugging Face Hub using 'huggingface-cli login'.")


if __name__ == "__main__":
    download_and_place_dataset()