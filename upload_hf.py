"""Utility script to upload the project to Hugging Face Spaces."""

import os
from huggingface_hub import HfApi

TOKEN = os.getenv("HF_TOKEN")
if not TOKEN:
    raise ValueError("Set HF_TOKEN environment variable before running this script.")

print("Initializing Hugging Face API...")
api = HfApi()

print("Uploading files to Space...")
try:
    api.upload_folder(
        folder_path=os.path.dirname(os.path.abspath(__file__)),
        repo_id="Deep9973/Enterprise-Guardian",
        repo_type="space",
        token=TOKEN,
        ignore_patterns=[".git", "__pycache__", "upload_hf.py", ".DS_Store"],
    )
    print("Upload completed successfully!")
except Exception as e:
    print(f"Failed to upload: {e}")
