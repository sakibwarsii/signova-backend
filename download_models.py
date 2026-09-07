import urllib.request
import os

os.makedirs('models', exist_ok=True)

models = {
    "hi_IN-pratham-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx",
    "hi_IN-priyamvada-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx",
    "hi_IN-rohan-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx",
    "mr_IN-google-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/mr/mr_IN/google/medium/mr_IN-google-medium.onnx",
    "ml_IN-arjun-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ml/ml_IN/arjun/medium/ml_IN-arjun-medium.onnx",
    "ml_IN-meera-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ml/ml_IN/meera/medium/ml_IN-meera-medium.onnx",
    "te_IN-maya-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/te/te_IN/maya/medium/te_IN-maya-medium.onnx",
    "te_IN-venkatesh-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/te/te_IN/venkatesh/medium/te_IN-venkatesh-medium.onnx",
}

for voice_id, base_url in models.items():
    model_path = f"models/{voice_id}.onnx"
    config_path = f"models/{voice_id}.onnx.json"
    
    if not os.path.exists(model_path):
        print(f"Downloading {voice_id} model...")
        urllib.request.urlretrieve(base_url, model_path)
    else:
        print(f"{voice_id} model already exists.")
        
    if not os.path.exists(config_path):
        print(f"Downloading {voice_id} config...")
        urllib.request.urlretrieve(base_url + ".json", config_path)
    else:
        print(f"{voice_id} config already exists.")

print("All downloads complete.")
