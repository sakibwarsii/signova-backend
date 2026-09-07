import json
import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from nlp_pipeline import smart_clean_text, process_text, translate_text_to_language
from tts_generator import generate_tts_base64
from vosk import Model, KaldiRecognizer
from ai_tools import ai_tools_router

app = FastAPI()

@app.on_event("startup")
async def warm_up_services():
    """
    Fire-and-forget background warm-up — NOT awaited, so it never adds to
    server boot time (which was already a known complaint). Its only job is
    to pay the one-time cold-start costs (edge-tts's first connection, Groq's
    first TLS/connection handshake) in the few seconds after boot instead of
    during a live demo's actual first spoken sentence, which is where that
    cost was showing up as a "huge delay" before.

    Piper/ONNX warm-up used to live here too. Removed: the .onnx voice model
    files (~496MB) are gitignored and nothing in the deploy pipeline
    downloads them, so on Railway they never exist and every Piper call was
    silently failing — see generate_tts_base64 below, which now handles
    every language via edge-tts instead (real neural voices exist for it in
    hi-IN, mr-IN, ml-IN, te-IN, not just en-US).
    """
    async def _warm():
        try:
            await generate_tts_base64("warm up", "en-US-AriaNeural")
            print("[Warmup] edge-tts connection primed.")
        except Exception as e:
            print(f"[Warmup] edge-tts warm-up failed (non-fatal): {e}")
        try:
            await smart_clean_text("warm up")
            print("[Warmup] Groq connection primed.")
        except Exception as e:
            print(f"[Warmup] Groq warm-up failed (non-fatal): {e}")
    asyncio.create_task(_warm())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_tools_router)

# Initialize Vosk Model
MODEL_PATH = "model" # vosk-model-small-en-in
if os.path.exists(MODEL_PATH):
    vosk_model = Model(MODEL_PATH)
else:
    vosk_model = None

# Load Dictionaries
SIGN_DICT_PATH = "sign_dict.json"
FINGERSPELL_PATH = "fingerspell.json"

sign_dict = {}
fingerspell_dict = {}

def load_dicts():
    global sign_dict, fingerspell_dict
    if os.path.exists(SIGN_DICT_PATH):
        with open(SIGN_DICT_PATH, 'r') as f:
            sign_dict = json.load(f)
    if os.path.exists(FINGERSPELL_PATH):
        with open(FINGERSPELL_PATH, 'r') as f:
            fingerspell_dict = json.load(f)

load_dicts()

def get_sigml_for_word(word: str) -> list[str]:
    if word in sign_dict:
        return [sign_dict[word]]
    sigml_list = []
    for char in word:
        if char in fingerspell_dict:
            sigml_list.append(fingerspell_dict[char])
    return sigml_list

async def transcribe_audio_groq(pcm_bytes: bytes) -> str:
    """Ultra-high accuracy multilingual speech transcription via Groq Whisper large-v3-turbo."""
    if not pcm_bytes or len(pcm_bytes) < 3200:
        return ""
    try:
        from groq_client import _primary_client, _fallback_client
        client = _primary_client or _fallback_client
        if not client:
            return ""
        import io, wave
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(pcm_bytes)
        buf.seek(0)
        buf.name = 'speech.wav'
        res = await client.audio.transcriptions.create(
            file=buf,
            model='whisper-large-v3-turbo',
            response_format='json'
        )
        return res.text.strip() if hasattr(res, 'text') else ""
    except Exception as e:
        print(f"[Groq Whisper] error: {e}")
        return ""

class ConnectionManager:
    def __init__(self):
        self.display_connections: list[WebSocket] = []

    async def connect_display(self, websocket: WebSocket):
        await websocket.accept()
        self.display_connections.append(websocket)

    def disconnect_display(self, websocket: WebSocket):
        if websocket in self.display_connections:
            self.display_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.display_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect_display(connection)

manager = ConnectionManager()

@app.websocket("/ws/display")
async def websocket_display(websocket: WebSocket):
    await manager.connect_display(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_display(websocket)

@app.websocket("/ws/teacher")
async def websocket_teacher(websocket: WebSocket):
    await websocket.accept()
    
    rec = KaldiRecognizer(vosk_model, 16000) if vosk_model else None
    audio_buffer = bytearray()
    
    # State for Delta Glossing (0-Lag)
    session_sent_glosses = []
    last_processed_word_count = 0
    current_target_language = "English"
    current_target_voice = "hi-IN-MadhurNeural"
    tts_enabled = False
    
    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "config":
                        current_target_language = data.get("targetLanguage", current_target_language)
                        current_target_voice = data.get("targetVoice", current_target_voice)
                        tts_enabled = data.get("ttsEnabled", tts_enabled)
                        continue

                    if data.get("type") in ("flush", "stop"):
                        vosk_text = ""
                        if rec is not None:
                            result = json.loads(rec.FinalResult())
                            vosk_text = result.get("text", "")
                        
                        whisper_text = ""
                        if len(audio_buffer) >= 3200:
                            whisper_text = await transcribe_audio_groq(bytes(audio_buffer))
                            audio_buffer.clear()
                        
                        raw_text = whisper_text or vosk_text
                        if raw_text:
                            current_sent_glosses = list(session_sent_glosses)
                            session_sent_glosses.clear()
                            last_processed_word_count = 0
                            asyncio.create_task(process_final_vosk(raw_text, current_sent_glosses, current_target_language, current_target_voice, tts_enabled))
                        continue
                        
                    if data.get("type") == "text":
                        raw_text = data.get("payload", "")
                        is_final = data.get("isFinal", False)
                        
                        words = raw_text.split()
                        
                        if not is_final:
                            # Print raw subtitles instantly
                            await manager.broadcast({"type": "partial_text_only", "text": raw_text})
                            
                            # Delta chunking every 5 words (Fast SpaCy ONLY)
                            if len(words) - last_processed_word_count >= 5:
                                glosses = process_text(raw_text)
                                
                                delta_glosses = glosses[len(session_sent_glosses):]
                                if delta_glosses:
                                    sigml_sequence = []
                                    for g in delta_glosses:
                                        sigml_sequence.extend(get_sigml_for_word(g))
                                    
                                    await manager.broadcast({
                                        "type": "partial_sigml",
                                        "glosses": delta_glosses,
                                        "sigml": sigml_sequence
                                    })
                                    session_sent_glosses.extend(delta_glosses)
                                last_processed_word_count = len(words)
                        
                        else:
                            # Final flush - Spawns Async Task to avoid blocking the loop
                            current_sent_glosses = list(session_sent_glosses)
                            session_sent_glosses.clear()
                            last_processed_word_count = 0
                            
                            async def process_final(text, sent_glosses, target_lang, target_voice, tts_on):
                                # Smart clean with Groq AI: removes stutters, filler words, fixes grammar/homophones,
                                # and translates regional languages into clean English for accurate sign dictionary matching.
                                cleaned_english = await smart_clean_text(text)
                                if not cleaned_english or len(cleaned_english.strip()) == 0:
                                    cleaned_english = text
                                
                                glosses = process_text(cleaned_english)
                                
                                delta_glosses = glosses[len(sent_glosses):]
                                sigml_sequence = []
                                for g in delta_glosses:
                                    sigml_sequence.extend(get_sigml_for_word(g))
                                    
                                translated_text = cleaned_english
                                if target_lang != "English":
                                    if any(ord(c) > 127 for c in text):
                                        translated_text = text
                                    else:
                                        translated_text = await translate_text_to_language(cleaned_english, target_lang)

                                audio_b64 = ""
                                if tts_on:
                                    audio_b64 = await generate_tts_base64(translated_text, target_voice)
                                
                                await manager.broadcast({
                                    "type": "final",
                                    "text": translated_text,
                                    "original_text": cleaned_english,
                                    "glosses": delta_glosses,
                                    "sigml": sigml_sequence,
                                    "audio": audio_b64
                                })
                            
                            asyncio.create_task(process_final(raw_text, current_sent_glosses, current_target_language, current_target_voice, tts_enabled))
                            
                except json.JSONDecodeError:
                    pass
                        
            # 2. AUDIO STREAMING MODE (Vosk + Groq Whisper Hybrid)
            elif "bytes" in message:
                data = message["bytes"]
                audio_buffer.extend(data)
                if len(audio_buffer) > 480000:
                    audio_buffer = audio_buffer[-480000:]

                if rec is not None and rec.AcceptWaveform(data):
                    vosk_text = json.loads(rec.Result()).get("text", "")
                    
                    whisper_text = ""
                    if len(audio_buffer) >= 3200:
                        whisper_text = await transcribe_audio_groq(bytes(audio_buffer))
                        audio_buffer.clear()
                    
                    raw_text = whisper_text or vosk_text
                    if raw_text:
                        current_sent_glosses = list(session_sent_glosses)
                        session_sent_glosses.clear()
                        last_processed_word_count = 0
                        
                        async def process_final_vosk(text, sent_glosses, target_lang, target_voice, tts_on):
                            cleaned_english = await smart_clean_text(text)
                            if not cleaned_english:
                                cleaned_english = text
                            glosses = process_text(cleaned_english)
                            
                            delta_glosses = glosses[len(sent_glosses):]
                            sigml_sequence = []
                            for g in delta_glosses:
                                sigml_sequence.extend(get_sigml_for_word(g))
                                
                            translated_text = cleaned_english
                            if target_lang != "English":
                                translated_text = await translate_text_to_language(cleaned_english, target_lang)

                            audio_b64 = ""
                            if tts_on:
                                audio_b64 = await generate_tts_base64(translated_text, target_voice)
                            
                            await manager.broadcast({
                                "type": "final",
                                "text": translated_text,
                                "original_text": cleaned_english,
                                "glosses": delta_glosses,
                                "sigml": sigml_sequence,
                                "audio": audio_b64
                            })
                            
                        asyncio.create_task(process_final_vosk(raw_text, current_sent_glosses, current_target_language, current_target_voice, tts_enabled))
                elif rec is not None:
                    partial = json.loads(rec.PartialResult())
                    raw_text = partial.get("partial", "")
                    if raw_text:
                        await manager.broadcast({"type": "partial_text_only", "text": raw_text})
                        
                        words = raw_text.split()
                        if len(words) - last_processed_word_count >= 5:
                            glosses = process_text(raw_text)
                            
                            delta_glosses = glosses[len(session_sent_glosses):]
                            if delta_glosses:
                                sigml_sequence = []
                                for g in delta_glosses:
                                    sigml_sequence.extend(get_sigml_for_word(g))
                                
                                await manager.broadcast({
                                    "type": "partial_sigml",
                                    "glosses": delta_glosses,
                                    "sigml": sigml_sequence
                                })
                                session_sent_glosses.extend(delta_glosses)
                            last_processed_word_count = len(words)

    except WebSocketDisconnect:
        print("Teacher disconnected")

if __name__ == "__main__":
    import uvicorn
    # Railway (and most hosts) inject the actual port to bind via $PORT —
    # hardcoding 8000 works locally but silently fails to receive traffic
    # on a host that expects a different port. reload=True is a dev-only
    # feature (re-imports on file change) that has no place in production;
    # RAILWAY_ENVIRONMENT is set automatically on Railway, so this only
    # disables it there — local `python main.py` is unaffected.
    port = int(os.environ.get("PORT", 8000))
    is_deployed = bool(os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RENDER"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=not is_deployed)
