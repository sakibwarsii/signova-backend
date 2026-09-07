import spacy
import os
from dotenv import load_dotenv
from groq_client import groq_chat_completion

load_dotenv()  # loads GROQ_API_KEY / GROQ_API_KEY_FALLBACK from .env file

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading SpaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

async def smart_clean_text(raw_text: str) -> str:
    """
    Advanced Intent Extractor for Indian Classrooms.
    Retains full vocabulary power but removes hesitations and stutters.
    """
    if not raw_text or len(raw_text.strip()) < 2:
        return ""
        
    # SAFETY-CRITICAL FIX: instruction #2 used to say "PREDICT and FILL IN any
    # skipped words to make the sentence grammatically complete and coherent"
    # — that's an explicit license for the model to INVENT content the teacher
    # never said. This is what turned "what are you doing priya" into a
    # completely different sentence, "why are you scared priya" — the model
    # decided the raw STT "needed" fixing and confidently fabricated a
    # different, coherent-sounding replacement. For a live interpreter, a
    # fabricated sentence is far worse than a rough or slightly garbled one:
    # it's not a transcription error, it's putting words in the teacher's
    # mouth that were never spoken. Replaced with an explicit ban on inventing
    # content — the model may only tidy up what's actually there.
    prompt = f"""
    You are an expert Educational Interpreter cleaning up a teacher's live speech
    for real-time sign-language interpretation. Accuracy matters more than
    polish — a rough-but-faithful sentence is far better than a smooth but
    WRONG one, because this output gets signed and spoken to students as if
    it's exactly what the teacher said.

    The following is raw STT output, which may contain stutters, filler words
    ("umm", "like"), or repeated words from the speech recognizer.

    Raw Speech: "{raw_text}"

    Task:
    1. Strip out meaningless filler words, stutters, and accidental word repeats.
    2. Fix obvious grammar so it reads as a normal sentence — but do NOT add,
       remove, or guess at any actual content. If a word is genuinely unclear,
       keep it as transcribed rather than replacing it with a different word.
    3. TRANSLATE any Hindi or Hinglish words into proper English.
    4. DO NOT dumb down the academic vocabulary.
    5. NEVER invent new words, phrases, or meaning that isn't already present
       in the raw speech. The cleaned sentence must mean exactly what the raw
       speech meant — no more, no less.
    6. ONLY output the final cleaned English sentence. No intro, no quotes.
    """
    
    try:
        # SECOND BUG FOUND ALONGSIDE THE HALLUCINATION ONE: openai/gpt-oss-120b
        # is a reasoning model — it spends tokens on an internal "thinking"
        # pass before producing the actual answer, and those reasoning tokens
        # count against max_tokens too. With the old max_tokens=100, a single
        # sentence's reasoning could burn the ENTIRE budget (measured 76 of
        # 93 tokens on a 5-word sentence), leaving zero tokens left for the
        # real output — the API call "succeeds" but returns an empty string,
        # which would silently produce zero glosses/signs for that line.
        # reasoning_effort="low" cuts reasoning token usage dramatically
        # (verified: 76 -> 21 tokens on the same test sentence, and FASTER —
        # 0.19s -> 0.08s) and the raised max_tokens is a safety margin on top.
        completion = await groq_chat_completion(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
            reasoning_effort="low"
        )
        result = completion.choices[0].message.content.strip()
        # Defense in depth: if the model still somehow returns nothing, fall
        # back to the raw STT text rather than silently producing zero signs —
        # a rough/unclean line the teacher actually said beats total silence.
        return result if result else raw_text
    except Exception as e:
        print(f"Groq API error: {e}")
        return raw_text

async def fast_multilingual_pipeline(raw_text: str, spoken_language: str = "English") -> dict:
    """
    Unified Single-Pass AI Pipeline for all 6 languages (English, Hindi, Marathi, Malayalam, Telugu, Kannada).
    CRITICAL REQUIREMENT: DO NOT translate the spoken subtitle text!
    - Subtitle text MUST remain in the exact same spoken language and script (cleaned of fillers/stutters).
    - Sign glosses are extracted as simple English base words for Indian Sign Language animation.
    """
    if not raw_text or len(raw_text.strip()) < 2:
        return {"subtitle_text": "", "english_text": "", "sign_glosses": []}

    prompt = f"""
You are an expert Educational Sign Language Interpreter.
Input Spoken Speech: "{raw_text}"
Spoken Language: {spoken_language}

CRITICAL RULES:
1. DO NOT translate the subtitle into another language. If the speech is in {spoken_language} (Hindi, Marathi, Telugu, Malayalam, Kannada, or English), the "subtitle_text" MUST BE in the EXACT SAME spoken language and script. Clean up filler words ("um", "uh", "umm", "like", stutters) and format with appropriate punctuation, but NEVER translate it into English or any other language.
2. Extract simple, essential English base words ("sign_glosses") for Indian Sign Language gestures (e.g. hello, welcome, student, teacher, learn, science, earth, water, book, good, morning, gravity).
3. Provide a brief faithful "english_text" translation solely for sign dictionary fallback.

Output JSON ONLY with exact keys:
{{
  "subtitle_text": "cleaned sentence strictly in original spoken {spoken_language} without translation",
  "english_text": "faithful English translation",
  "sign_glosses": ["list", "of", "english", "base", "sign", "words"]
}}
"""
    try:
        import json
        completion = await groq_chat_completion(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        content = completion.choices[0].message.content.strip()
        data = json.loads(content)
        return {
            "subtitle_text": data.get("subtitle_text") or raw_text,
            "english_text": data.get("english_text") or raw_text,
            "sign_glosses": [g.lower() for g in data.get("sign_glosses", []) if isinstance(g, str)]
        }
    except Exception as e:
        print(f"[fast_multilingual_pipeline] Groq error: {e}")
        try:
            cleaned = await smart_clean_text(raw_text)
            glosses = process_text(cleaned)
            return {"subtitle_text": raw_text, "english_text": cleaned, "sign_glosses": glosses}
        except Exception:
            return {"subtitle_text": raw_text, "english_text": raw_text, "sign_glosses": [w.lower() for w in raw_text.split()]}

def process_text(text: str) -> list:
    doc = nlp(text.lower())
    glosses = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        if token.lemma_ in ['is', 'are', 'am', 'was', 'were', 'be', 'been', 'a', 'an', 'the']:
            continue
        glosses.append(token.lemma_)
    return glosses

async def generate_script_from_prompt(topic: str, instructions: str) -> str:
    prompt = f"""
    You are an expert Educational Content Creator. 
    Topic: {topic}
    Instructions and Required Words: {instructions}
    
    Task: Write a clear, easy-to-understand educational explanation suitable for students. 
    Keep it informative and engaging. Do not include any conversational filler, stage directions, or markdown.
    Write a concise script of approximately 50 to 100 words.
    """
    
    try:
        completion = await groq_chat_completion(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
            reasoning_effort="low"
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API error: {e}")
        return f"Sorry, I encountered an error generating the script: {str(e)}"

async def summarize_and_script_chunk(text_chunk: str) -> dict:
    prompt = f"""
    You are an expert Educational Interpreter.
    Extract the core educational concepts from the following text chunk and rewrite it into a clear, concise, easy-to-understand educational script suitable for students.
    Ignore any boilerplate, page numbers, formatting artifacts, or filler content.
    Keep the script very concise (under 60 words if possible) to optimize for sign language translation.
    
    You must ALSO provide a 1-3 word "visual_keyword" that describes the main visual subject of the text (e.g. "Apple Tree", "Addition", "Thirsty Crow", "Physics"). This will be used to fetch a background image.

    Output a valid JSON object with EXACTLY two keys: "script" and "visual_keyword". Do not include any other text.

    Text Chunk:
    {text_chunk}
    """
    
    try:
        completion = await groq_chat_completion(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500,
            reasoning_effort="low",
            response_format={"type": "json_object"}
        )
        import json
        return json.loads(completion.choices[0].message.content.strip())
    except Exception as e:
        print(f"Groq API error: {e}")
        return {"script": "", "visual_keyword": ""}

async def translate_text_to_language(text: str, target_language: str) -> str:
    """
    Translates educational text to the target language.
    """
    if not text or len(text.strip()) == 0:
        return ""
        
    prompt = f"""
    You are an expert Educational Translator.
    Translate the following English text to {target_language}.
    Keep the educational tone and accuracy.
    ONLY output the translated text. No quotes, no explanations, no original text.
    
    English Text: "{text}"
    """
    
    try:
        completion = await groq_chat_completion(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
            reasoning_effort="low"
        )
        result = completion.choices[0].message.content.strip()
        return result if result else text
    except Exception as e:
        print(f"Groq API Translation error: {e}")
        return text # fallback to original text
