import os
import urllib.request
import zipfile

def setup():
    if os.path.exists("model"):
        print("Vosk model already exists in 'model/' directory.")
        return

    print("Downloading Vosk model (en-in)...")
    url = "https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip"
    zip_path = "vosk-model.zip"
    
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting Vosk model...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
        
    if os.path.exists("vosk-model-small-en-in-0.4"):
        if os.path.exists("model"):
            import shutil
            shutil.rmtree("model")
        os.rename("vosk-model-small-en-in-0.4", "model")
        
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    print("Vosk model setup complete.")

if __name__ == "__main__":
    setup()
