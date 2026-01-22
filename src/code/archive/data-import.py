import requests
import os

"""DEPRACATED DO NOT USE"""

"""
API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
LABEL_STUDIO_URL = "http://localhost:8080"
PROJECT_ID = 1  # Replace with your project ID
IMAGE_DIR = "/images"  # The directory where your images are located within the container.

headers = {"Authorization": f"Token {API_KEY}"}

for filename in os.listdir(IMAGE_DIR):
    filepath = os.path.join(IMAGE_DIR, filename)
    if os.path.isfile(filepath):
        print(f"Importing {filepath}...")
        try:
            with open(filepath, "rb") as f:
                files = {"file": (filename, f)}
                response = requests.post(
                    f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/import",
                    headers=headers,
                    files=files,
                )
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                print(f"Imported {filename} successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error importing {filename}: {e}")
    else:
        print(f"Skipping {filepath}, not a file.")

print("Data import complete.")
"""