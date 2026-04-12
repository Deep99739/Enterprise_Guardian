import os
from huggingface_hub import HfApi

print("Initializing Hugging Face API...")
api = HfApi()

print("Uploading files to Space...")
try:
    api.upload_folder(
        folder_path="/Users/mayankkumar/Documents/Open_Env_hackathon/enterprise_guardian",
        repo_id="Deep9973/Enterprise-Guardian",
        repo_type="space",
        token="$(echo HF_TOKEN_PLACEHOLDER)",
        ignore_patterns=[".git", "__pycache__", "upload_hf.py", ".DS_Store"]
    )
    print("Upload completed successfully!")
except Exception as e:
    print(f"Failed to upload: {e}")
