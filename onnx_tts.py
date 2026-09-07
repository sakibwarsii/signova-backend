import os
import wave
import io
import base64
import asyncio
from piper.voice import PiperVoice

# Pre-load voices to avoid loading time on each request. 
# We'll support Hindi for now, but design it to be expandable.
loaded_voices = {}

def get_piper_voice(voice_id: str):
    global loaded_voices
    if voice_id in loaded_voices:
        return loaded_voices[voice_id]
    
    # Path configuration for models
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(model_dir, f"{voice_id}.onnx")
    config_path = os.path.join(model_dir, f"{voice_id}.onnx.json")
        
    if os.path.exists(model_path) and os.path.exists(config_path):
        print(f"Loading ONNX model for {voice_id}...")
        voice = PiperVoice.load(model_path, config_path)
        loaded_voices[voice_id] = voice
        return voice
    else:
        print(f"Model files not found for {voice_id}: {model_path}")
    return None

def _synthesize_sync(clean_text: str, voice) -> bytes:
    """CPU-bound ONNX inference + WAV encoding. Runs off the event loop via to_thread."""
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        for chunk in voice.synthesize(clean_text):
            wav_file.writeframes(chunk.audio_int16_bytes)
    return wav_io.getvalue()

async def generate_onnx_tts_base64(text: str, voice_id: str) -> str:
    """
    Generates TTS audio using local ONNX (Piper) and returns it as a base64 data URL.
    Synthesis is CPU-bound, so it's offloaded to a worker thread — otherwise it would
    block the entire asyncio event loop (freezing every other connection: other
    classrooms' mic streams, student displays, etc.) for the duration of inference.
    """
    if not text.strip():
        return ""

    clean_text = text.replace("<", "").replace(">", "").strip()
    if not clean_text:
        return ""

    voice = get_piper_voice(voice_id)
    if not voice:
        print(f"ONNX TTS: No local model found for {voice_id}")
        return ""

    try:
        audio_bytes = await asyncio.to_thread(_synthesize_sync, clean_text, voice)
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        return f"data:audio/wav;base64,{base64_audio}"
    except Exception as e:
        print(f"ONNX TTS generation error: {e}")
        return ""
