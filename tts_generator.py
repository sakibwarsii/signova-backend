import edge_tts
import base64
import tempfile
import os

async def generate_tts_base64(text: str, voice: str = "en-US-AriaNeural") -> str:
    """
    Generates TTS audio using edge-tts and returns it as a base64 data URL.
    """
    if not text.strip():
        return ""
        
    # Remove any tags or weird characters that might mess up TTS
    clean_text = text.replace("<", "").replace(">", "").strip()
    if not clean_text:
        return ""
        
    communicate = edge_tts.Communicate(clean_text, voice)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name
        
    try:
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
            
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        return f"data:audio/mp3;base64,{base64_audio}"
    except Exception as e:
        print(f"TTS generation error: {e}")
        return ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
