import os
from huggingface_hub import snapshot_download

model_name = "ASLP-lab/Easy-Turn"
local_dir = "Easy-Turn"  # Directory to save the downloaded model

try:
    # Create the directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Download the checkpoint model
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        repo_type="model",  # Specifies that we're downloading a model
        local_dir_use_symlinks=False # important for making the checkpoint portable
    )

    print(f"Model '{model_name}' downloaded successfully to '{local_dir}'.")

except Exception as e:
    print(f"Error downloading model: {e}")