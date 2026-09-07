import json
import os
import asyncio
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from nlp_pipeline import process_text, generate_script_from_prompt, summarize_and_script_chunk, nlp, translate_text_to_language
from document_processor import extract_text, chunk_text
from visual_assistant import plan_visuals, build_visual_library, annotate_chunks_with_visuals
from tts_generator import generate_tts_base64

ai_tools_router = APIRouter()

# Decoupled dictionary loading for safety
SIGN_DICT_PATH = "sign_dict.json"
FINGERSPELL_PATH = "fingerspell.json"
sign_dict = {}
fingerspell_dict = {}

if os.path.exists(SIGN_DICT_PATH):
    with open(SIGN_DICT_PATH, 'r') as f:
        sign_dict = json.load(f)
if os.path.exists(FINGERSPELL_PATH):
    with open(FINGERSPELL_PATH, 'r') as f:
        fingerspell_dict = json.load(f)

def get_sigml_for_word_ai(word: str) -> list[str]:
    if word in sign_dict:
        return [sign_dict[word]]
    sigml_list = []
    for char in word:
        if char in fingerspell_dict:
            sigml_list.append(fingerspell_dict[char])
    return sigml_list

class BatchTextRequest(BaseModel):
    text: str

def get_mini_chunks(text: str, max_words=5):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

async def generate_audio_for_chunks(chunks: list, lang: str, voice: str, tts: bool,
                                     progress_callback=None, concurrency: int = 4) -> list:
    """
    Fills chunk["audio_base64"] (and chunk["translated_text"] when translated) for every
    chunk, running translation + TTS concurrently (bounded by `concurrency`) instead of
    one chunk at a time. For N chunks this turns ~N sequential round-trips into a few
    concurrent batches — the dominant speedup for batch/prompt-to-sign generation time.
    """
    if not tts or not chunks:
        for c in chunks:
            c["audio_base64"] = ""
        return chunks

    sem = asyncio.Semaphore(concurrency)
    total = len(chunks)
    completed = 0

    async def process_one(chunk):
        nonlocal completed
        async with sem:
            try:
                text_for_audio = chunk["text"]
                if lang != "English":
                    translated_text = await translate_text_to_language(chunk["text"], lang)
                    chunk["translated_text"] = translated_text
                    text_for_audio = translated_text
                # All languages route through edge-tts — the old Piper/ONNX
                # path for non-English silently produced no audio at all in
                # production, since its model files were never actually
                # deployed (gitignored, ~496MB, no download step on Railway).
                # edge-tts has real neural voices for every language this
                # app supports, so there's no quality tradeoff.
                chunk["audio_base64"] = await generate_tts_base64(text_for_audio, voice)
            except Exception as e:
                print(f"[AudioGen] Error for chunk '{str(chunk.get('text',''))[:30]}': {e}")
                chunk["audio_base64"] = ""
            finally:
                completed += 1
                if progress_callback:
                    await progress_callback(completed, total)

    await asyncio.gather(*(process_one(c) for c in chunks))
    return chunks

@ai_tools_router.post("/api/batch-text")
async def batch_text_endpoint(req: BatchTextRequest, visuals: bool = False, lang: str = "English", voice: str = "en-US-AriaNeural", tts: bool = True):
    async def generate_response():
        try:
            yield json.dumps({"status": "Parsing text..."}) + "\n"
            doc = nlp(req.text)
            
            raw_chunks = []
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue
                mini_chunks = get_mini_chunks(sent_text)
                for mc in mini_chunks:
                    glosses = process_text(mc)
                    sigml_sequence = []
                    for g in glosses:
                        sigml_sequence.extend(get_sigml_for_word_ai(g))
                    if sigml_sequence:
                        raw_chunks.append({"text": mc, "sigml": sigml_sequence})
            
            # Annotate with visuals if requested
            if visuals and raw_chunks:
                yield json.dumps({"status": "Planning visuals..."}) + "\n"
                full_text = " ".join([c["text"] for c in raw_chunks])
                keyword_plan = await plan_visuals(full_text)
                
                progress_queue = asyncio.Queue()
                
                async def visual_progress(completed, total):
                    await progress_queue.put({"status": f"Generating {total} visuals ({completed}/{total})..."})
                
                yield json.dumps({"status": f"Generating {len(keyword_plan)} visuals (0/{len(keyword_plan)})..."}) + "\n"
                
                fetch_task = asyncio.create_task(build_visual_library(keyword_plan, progress_callback=visual_progress))
                while not fetch_task.done():
                    try:
                        msg = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                        yield json.dumps(msg) + "\n"
                    except asyncio.TimeoutError:
                        pass
                
                while not progress_queue.empty():
                    msg = progress_queue.get_nowait()
                    yield json.dumps(msg) + "\n"
                
                visual_library = fetch_task.result()
                raw_chunks = annotate_chunks_with_visuals(raw_chunks, visual_library)

            total_chunks = len(raw_chunks)
            if total_chunks:
                yield json.dumps({"status": f"Generating audio (0/{total_chunks})..."}) + "\n"
                audio_progress_queue = asyncio.Queue()

                async def audio_progress(completed, total):
                    await audio_progress_queue.put({"status": f"Generating audio ({completed}/{total})..."})

                audio_task = asyncio.create_task(
                    generate_audio_for_chunks(raw_chunks, lang, voice, tts, progress_callback=audio_progress)
                )
                while not audio_task.done():
                    try:
                        msg = await asyncio.wait_for(audio_progress_queue.get(), timeout=0.1)
                        yield json.dumps(msg) + "\n"
                    except asyncio.TimeoutError:
                        pass
                while not audio_progress_queue.empty():
                    yield json.dumps(audio_progress_queue.get_nowait()) + "\n"
                chunks = audio_task.result()
            else:
                chunks = []

            yield json.dumps({"status": "Done!"}) + "\n"
            yield json.dumps({"chunks": chunks}) + "\n"
        except Exception as e:
            print(f"Error in batch_text: {e}")
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(generate_response(), media_type="application/x-ndjson")

class PromptRequest(BaseModel):
    topic: str
    instructions: str

@ai_tools_router.post("/api/prompt-to-sign")
async def prompt_to_sign_endpoint(req: PromptRequest, visuals: bool = False, lang: str = "English", voice: str = "en-US-AriaNeural", tts: bool = True):
    async def generate_response():
        try:
            yield json.dumps({"status": "Generating script..."}) + "\n"
            script = await generate_script_from_prompt(req.topic, req.instructions)
            
            yield json.dumps({"status": "Parsing text..."}) + "\n"
            doc = nlp(script)
            
            raw_chunks = []
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue
                mini_chunks = get_mini_chunks(sent_text)
                for mc in mini_chunks:
                    glosses = process_text(mc)
                    sigml_sequence = []
                    for g in glosses:
                        sigml_sequence.extend(get_sigml_for_word_ai(g))
                    if sigml_sequence:
                        raw_chunks.append({"text": mc, "sigml": sigml_sequence})
            
            # Annotate with visuals if requested
            if visuals and raw_chunks:
                yield json.dumps({"status": "Planning visuals..."}) + "\n"
                full_text = " ".join([c["text"] for c in raw_chunks])
                keyword_plan = await plan_visuals(full_text)
                
                progress_queue = asyncio.Queue()
                
                async def visual_progress(completed, total):
                    await progress_queue.put({"status": f"Generating {total} visuals ({completed}/{total})..."})
                
                yield json.dumps({"status": f"Generating {len(keyword_plan)} visuals (0/{len(keyword_plan)})..."}) + "\n"
                
                fetch_task = asyncio.create_task(build_visual_library(keyword_plan, progress_callback=visual_progress))
                while not fetch_task.done():
                    try:
                        msg = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                        yield json.dumps(msg) + "\n"
                    except asyncio.TimeoutError:
                        pass
                
                while not progress_queue.empty():
                    msg = progress_queue.get_nowait()
                    yield json.dumps(msg) + "\n"
                
                visual_library = fetch_task.result()
                raw_chunks = annotate_chunks_with_visuals(raw_chunks, visual_library)

            total_chunks = len(raw_chunks)
            if total_chunks:
                yield json.dumps({"status": f"Generating audio (0/{total_chunks})..."}) + "\n"
                audio_progress_queue = asyncio.Queue()

                async def audio_progress(completed, total):
                    await audio_progress_queue.put({"status": f"Generating audio ({completed}/{total})..."})

                audio_task = asyncio.create_task(
                    generate_audio_for_chunks(raw_chunks, lang, voice, tts, progress_callback=audio_progress)
                )
                while not audio_task.done():
                    try:
                        msg = await asyncio.wait_for(audio_progress_queue.get(), timeout=0.1)
                        yield json.dumps(msg) + "\n"
                    except asyncio.TimeoutError:
                        pass
                while not audio_progress_queue.empty():
                    yield json.dumps(audio_progress_queue.get_nowait()) + "\n"
                chunks = audio_task.result()
            else:
                chunks = []

            yield json.dumps({"status": "Done!"}) + "\n"
            yield json.dumps({"chunks": chunks}) + "\n"
        except Exception as e:
            print(f"Error in prompt_to_sign: {e}")
            yield json.dumps({"error": str(e)}) + "\n"
            
    return StreamingResponse(generate_response(), media_type="application/x-ndjson")

@ai_tools_router.post("/api/upload-document")
async def upload_document_endpoint(file: UploadFile = File(...), visuals: bool = False, lang: str = "English", voice: str = "en-US-AriaNeural", tts: bool = True):
    async def generate_chunks():
        try:
            # Extract text from the uploaded document, limiting to 5000 words
            raw_text = await extract_text(file, max_words=5000)
            
            # Pre-plan visuals from full text BEFORE streaming begins (smarter)
            visual_library = {}
            if visuals:
                keyword_plan = await plan_visuals(raw_text)
                visual_library = await build_visual_library(keyword_plan)
            
            # Chunk the raw text into sections of 250 words
            text_chunks = chunk_text(raw_text, chunk_size_words=250)
            
            for section in text_chunks:
                # Ask LLM to summarize section into a signable script
                llm_response = await summarize_and_script_chunk(section)
                script = llm_response.get("script", "")

                if not script:
                    continue

                # Parse the script into sentences and sign glosses
                doc = nlp(script)
                section_chunks = []
                for sent in doc.sents:
                    sent_text = sent.text.strip()
                    if not sent_text:
                        continue

                    mini_chunks = get_mini_chunks(sent_text)
                    for mc in mini_chunks:
                        glosses = process_text(mc)
                        sigml_sequence = []
                        for g in glosses:
                            sigml_sequence.extend(get_sigml_for_word_ai(g))

                        if sigml_sequence:
                            section_chunks.append({"text": mc, "sigml": sigml_sequence})

                if not section_chunks:
                    continue

                # Translate + synthesize every mini-chunk in this section concurrently
                # instead of one Groq round-trip + Piper synth at a time — this is the
                # dominant cost of document upload (a section can have dozens of
                # mini-chunks, each previously a fully serial network call).
                await generate_audio_for_chunks(section_chunks, lang, voice, tts)

                for chunk_data in section_chunks:
                    # Enrich with pre-planned visual library
                    if visual_library:
                        annotated = annotate_chunks_with_visuals([chunk_data], visual_library)
                        chunk_data = annotated[0]

                    # Yield chunk as a JSON string with a newline delimiter
                    yield json.dumps(chunk_data) + "\n"
        except Exception as e:
            print(f"Error processing document: {e}")
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(generate_chunks(), media_type="application/x-ndjson")

@ai_tools_router.post("/api/test-pixazo")
async def test_pixazo_endpoint(req: BatchTextRequest):
    import httpx
    try:
        PIXAZO_API_KEY = os.environ.get("PIXAZO_API_KEY", "19e791b8f7c247a991324e66ef1e0ab6")
        pixazo_url = "https://gateway.pixazo.ai/flux-1-schnell/v1/getData"
        headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "Ocp-Apim-Subscription-Key": PIXAZO_API_KEY
        }
        payload = {
            "prompt": req.text,
            "num_steps": 1,
            "height": 512,
            "width": 512
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(pixazo_url, json=payload, headers=headers, timeout=30.0)
            
            return {
                "status": resp.status_code,
                "response": resp.json() if resp.status_code == 200 else resp.text
            }
    except Exception as e:
        return {"error": str(e)}
