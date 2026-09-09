import json
import os
import re
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from nlp_pipeline import smart_clean_text, process_text, translate_text_to_language, fast_multilingual_pipeline
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
    clean_w = re.sub(r'[^a-zA-Z0-9]', '', word).lower().strip()
    if not clean_w:
        return []
    if clean_w in sign_dict:
        return [sign_dict[clean_w]]
    sigml_list = []
    for char in clean_w:
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
            response_format='json',
            prompt="Indian English classroom teacher speaking with Indian accent. Common words: working, work, walking, science, physics, mathematics, students, computer, education."
        )
        return res.text.strip() if hasattr(res, 'text') else ""
    except Exception as e:
        print(f"[Groq Whisper] error: {e}")
        return ""

import uuid
import re
import time

class ConnectionManager:
    def __init__(self):
        # Maps session_id -> list of display WebSocket connections for complete multi-user isolation
        from collections import defaultdict
        self.session_displays: dict[str, list[WebSocket]] = defaultdict(list)
        # Cache of recent utterances per session to prevent repeated speech/signs: session_id -> list of (timestamp, norm_text)
        self.session_history: dict[str, list[tuple[float, str]]] = defaultdict(list)

    async def connect_display(self, websocket: WebSocket, session_id: str = ""):
        await websocket.accept()
        sess = session_id.strip() if session_id and session_id.strip() and session_id.strip() != "default" else f"auto_{uuid.uuid4().hex[:10]}"
        self.session_displays[sess].append(websocket)
        print(f"[WS Display] Connected to session: {sess} (total displays in session: {len(self.session_displays[sess])})")
        return sess

    def disconnect_display(self, websocket: WebSocket, session_id: str):
        if session_id in self.session_displays and websocket in self.session_displays[session_id]:
            self.session_displays[session_id].remove(websocket)
        if session_id in self.session_displays and not self.session_displays[session_id]:
            self.session_displays.pop(session_id, None)
            self.session_history.pop(session_id, None)
        print(f"[WS Display] Disconnected from session: {session_id}")

    def is_duplicate(self, session_id: str, text: str) -> bool:
        """Drops repeated utterances within 8 seconds for the same session to prevent duplicate speech/signing."""
        norm = re.sub(r'[\s.,/#!$%^&*;:{}=\-_`~()\'\"।?]+', '', text.lower().strip())
        if not norm:
            return True
        now = time.time()
        # Keep utterances from last 3.5 seconds to prevent accidental double-delivery
        history = [item for item in self.session_history[session_id] if now - item[0] < 3.5]
        for t, old_norm in history:
            if old_norm == norm:
                return True
        history.append((now, norm))
        self.session_history[session_id] = history[-10:]
        return False

    async def send_to_session(self, session_id: str, message: dict):
        """Broadcasts messages strictly to display connections belonging to the sender's session."""
        displays = list(self.session_displays.get(session_id, []))
        for connection in displays:
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect_display(connection, session_id)

manager = ConnectionManager()

@app.websocket("/ws/display")
async def websocket_display(websocket: WebSocket, session_id: str = ""):
    active_session = await manager.connect_display(websocket, session_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_display(websocket, active_session)

@app.websocket("/ws/teacher")
async def websocket_teacher(websocket: WebSocket, session_id: str = ""):
    await websocket.accept()
    # Ensure every teacher has a valid session id matching its display socket
    active_session = session_id.strip() if session_id and session_id.strip() and session_id.strip() != "default" else f"auto_{uuid.uuid4().hex[:10]}"
    print(f"[WS Teacher] Connected with session: {active_session}")
    
    rec = KaldiRecognizer(vosk_model, 16000) if vosk_model else None
    audio_buffer = bytearray()
    
    current_target_language = "English"
    current_target_voice = "hi-IN-MadhurNeural"
    tts_enabled = False
    
    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            async def handle_final_utterance(text: str, spoken_lang: str, target_voice: str, tts_on: bool, target_sess: str):
                try:
                    clean_raw = text.strip()
                    if not clean_raw:
                        return
                    if manager.is_duplicate(target_sess, clean_raw):
                        print(f"[Deduplication] Dropped duplicate utterance for {target_sess}: '{clean_raw}'")
                        return

                    # 100% VERBATIM SUBTITLE - NEVER TRANSLATED!
                    verbatim_subtitle = clean_raw
                    
                    pipeline_res = await fast_multilingual_pipeline(clean_raw, spoken_lang)
                    cleaned_english = pipeline_res.get("english_text") or clean_raw
                    glosses = pipeline_res.get("sign_glosses") or [w.lower() for w in cleaned_english.split()]

                    sigml_sequence = []
                    for g in glosses:
                        sigml_sequence.extend(get_sigml_for_word(g))

                    audio_b64 = ""
                    if tts_on:
                        try:
                            audio_b64 = await generate_tts_base64(verbatim_subtitle, target_voice)
                        except Exception as e:
                            print(f"[TTS] error: {e}")

                    await manager.send_to_session(target_sess, {
                        "type": "final",
                        "text": verbatim_subtitle,
                        "original_text": cleaned_english,
                        "glosses": glosses,
                        "sigml": sigml_sequence,
                        "audio": audio_b64
                    })
                except Exception as e:
                    print(f"[handle_final_utterance] error: {e}")

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    # Check if client passed an explicit session_id in the JSON payload
                    msg_sess = data.get("session_id") or active_session
                    
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
                            asyncio.create_task(handle_final_utterance(raw_text, current_target_language, current_target_voice, tts_enabled, msg_sess))
                        continue
                        
                    if data.get("type") == "text":
                        raw_text = data.get("payload", "")
                        is_final = data.get("isFinal", False)
                        spoken_lang = data.get("language") or current_target_language
                        
                        if not is_final:
                            # Stream raw partial subtitles instantly for zero-latency visual feedback
                            await manager.send_to_session(msg_sess, {"type": "partial_text_only", "text": raw_text})
                        else:
                            # Final utterance: process clean verbatim subtitles in spoken language & ISL signs
                            asyncio.create_task(handle_final_utterance(raw_text, spoken_lang, current_target_voice, tts_enabled, msg_sess))
                            
                except json.JSONDecodeError:
                    pass
                        
            # 2. AUDIO STREAMING MODE (Vosk + Groq Whisper Hybrid)
            elif "bytes" in message:
                data = message["bytes"]
                audio_buffer.extend(data)
                if len(audio_buffer) > 480000:
                    audio_buffer = audio_buffer[-480000:]

                # For Indian languages, Groq Whisper transcribes speech when buffer reaches 1.5 seconds or silence
                if rec is not None and rec.AcceptWaveform(data):
                    vosk_text = json.loads(rec.Result()).get("text", "")
                    
                    whisper_text = ""
                    if len(audio_buffer) >= 3200:
                        whisper_text = await transcribe_audio_groq(bytes(audio_buffer))
                        audio_buffer.clear()
                    
                    raw_text = whisper_text or vosk_text
                    if raw_text:
                        asyncio.create_task(handle_final_utterance(raw_text, current_target_language, current_target_voice, tts_enabled, active_session))
                elif rec is not None:
                    partial = json.loads(rec.PartialResult())
                    raw_text = partial.get("partial", "")
                    if raw_text:
                        await manager.send_to_session(active_session, {"type": "partial_text_only", "text": raw_text})
                elif rec is None:
                    # Cloud/Render streaming: Groq Whisper transcribes when buffer reaches ~1.5s of audio (48000 bytes)
                    if len(audio_buffer) >= 48000:
                        buf_to_transcribe = bytes(audio_buffer)
                        audio_buffer.clear()
                        whisper_text = await transcribe_audio_groq(buf_to_transcribe)
                        if whisper_text and whisper_text.strip():
                            asyncio.create_task(handle_final_utterance(whisper_text.strip(), current_target_language, current_target_voice, tts_enabled, active_session))

    except WebSocketDisconnect:
        print("Teacher disconnected")

@app.post("/api/transcribe-audio")
async def transcribe_audio_endpoint(file: UploadFile = File(...), lang: str = "English", session_id: str = "default"):
    """
    Direct endpoint for casting / video audio to sign conversion:
    Transcribes the uploaded audio segment using Groq Whisper, converts the text into
    Indian Sign Language (ISL) glosses and SiGML animation sequence, broadcasts the
    signs to the active classroom session display, and returns the response directly.
    """
    try:
        audio_content = await file.read()
        if not audio_content or len(audio_content) < 1000:
            return {"text": "", "sigml": []}

        from groq_client import _primary_client, _fallback_client
        client = _primary_client or _fallback_client
        if not client:
            return {"error": "Groq client not available", "text": "", "sigml": []}

        import io
        buf = io.BytesIO(audio_content)
        buf.name = file.filename or "audio.webm"

        res = await client.audio.transcriptions.create(
            file=buf,
            model="whisper-large-v3-turbo",
            response_format="json",
            prompt="Indian English speech with Indian accent. Common words: working, work, walking, science, physics, mathematics, students, education."
        )
        raw_text = res.text.strip() if hasattr(res, "text") else ""
        if not raw_text:
            return {"text": "", "sigml": []}

        # Deduplicate to prevent double-processing identical audio slices
        target_sess = session_id.strip() if session_id and session_id.strip() else "default"
        if manager.is_duplicate(target_sess, raw_text):
            return {"text": raw_text, "sigml": [], "duplicate": True}

        # Convert text into sign glosses via fast multilingual pipeline
        pipeline_res = await fast_multilingual_pipeline(raw_text, lang)
        cleaned_english = pipeline_res.get("english_text") or raw_text
        glosses = pipeline_res.get("sign_glosses") or [w.lower() for w in cleaned_english.split()]

        sigml_sequence = []
        for g in glosses:
            sigml_sequence.extend(get_sigml_for_word(g))

        payload = {
            "type": "final",
            "text": raw_text,
            "original_text": cleaned_english,
            "glosses": glosses,
            "sigml": sigml_sequence,
            "source": "cast_video"
        }

        # Broadcast directly to session displays so classroom avatar signs in real time
        await manager.send_to_session(target_sess, payload)

        return payload
    except Exception as e:
        print(f"[transcribe-audio] error: {e}")
        return {"error": str(e), "text": "", "sigml": []}

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
