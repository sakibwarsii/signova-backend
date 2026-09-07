"""
Prepends a short "Good morning, class!" greeting (in the demo's own language)
to the front of each existing Quick Demo file, so the avatar opens with a
warm intro before launching into the topic — instead of jumping straight in.
No visuals needed for the greeting itself, keeps it lightweight.
"""
import asyncio
import json
import sys

sys.path.insert(0, ".")
from nlp_pipeline import nlp, process_text
from ai_tools import get_sigml_for_word_ai, get_mini_chunks, generate_audio_for_chunks

DEMO_DIR = "F:/speech to sign(updated)/web/public/demo_samples"


async def build_greeting_chunks(greeting_text: str):
    doc = nlp(greeting_text)
    raw_chunks = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        for mc in get_mini_chunks(sent_text):
            glosses = process_text(mc)
            sigml_sequence = []
            for g in glosses:
                sigml_sequence.extend(get_sigml_for_word_ai(g))
            if sigml_sequence:
                raw_chunks.append({"text": mc, "sigml": sigml_sequence})
    return raw_chunks


async def prepend_greeting(demo_file: str, greeting_text: str, lang: str, voice: str):
    raw_chunks = await build_greeting_chunks(greeting_text)
    await generate_audio_for_chunks(raw_chunks, lang, voice, True)

    path = f"{DEMO_DIR}/{demo_file}"
    with open(path, encoding="utf-8") as f:
        existing = json.load(f)

    # Guard against re-running this script twice on an already-greeted file —
    # detect by checking if the first chunk's english text already matches
    # our greeting opener.
    if existing and existing[0].get("text", "").lower().startswith("good morning"):
        print(f"  {demo_file}: already has a greeting, replacing it")
        # Find where the original content starts again by dropping chunks
        # that came from THIS greeting text (same count as raw_chunks would
        # have been last time) is fragile across edits, so instead just
        # detect the boundary: greeting chunks never carry a visual_url,
        # first chunk WITH a visual_url (or, for chunks with none at all,
        # first chunk not exactly matching our current greeting's first
        # mini-chunk) marks where the real content begins. Simplest safe
        # approach: re-derive by dropping a leading run of chunks with no
        # visual_url AND no audio-based marker — since real content in these
        # demos always eventually has a visual_url except a few chunks, this
        # single script is only ever run once per file in practice, so just
        # skip files that already look greeted rather than guessing a splice.
        print(f"  Skipping {demo_file} (already greeted) — delete its greeting manually first if you want to regenerate it.")
        return

    combined = raw_chunks + existing
    with open(path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False)
    print(f"  -> {demo_file}: prepended {len(raw_chunks)} greeting chunks ({len(combined)} total)")


async def main():
    jobs = [
        ("photosynthesis_en.json", "Good morning, class! Today we will learn about photosynthesis.", "English", "en-US-AriaNeural"),

        ("ohms_law_mr.json", "Good morning, class! Today we will learn about Ohm's Law.", "Marathi", "mr_IN-google-medium"),
        ("ohms_law_en.json", "Good morning, class! Today we will learn about Ohm's Law.", "English", "en-US-AriaNeural"),

        ("thirsty_crow_hi.json", "Good morning, class! Today we will hear a short story, the thirsty crow.", "Hindi", "hi_IN-priyamvada-medium"),
        ("thirsty_crow_en.json", "Good morning, class! Today we will hear a short story, the thirsty crow.", "English", "en-US-AriaNeural"),

        ("profit_and_loss_te.json", "Good morning, class! Today we will learn about profit and loss.", "Telugu", "te_IN-maya-medium"),
        ("profit_and_loss_hi.json", "Good morning, class! Today we will learn about profit and loss.", "Hindi", "hi_IN-priyamvada-medium"),
        ("profit_and_loss_en.json", "Good morning, class! Today we will learn about profit and loss.", "English", "en-US-AriaNeural"),

        ("supply_and_demand_ml.json", "Good morning, class! Today we will learn about supply and demand.", "Malayalam", "ml_IN-meera-medium"),
        ("supply_and_demand_hi.json", "Good morning, class! Today we will learn about supply and demand.", "Hindi", "hi_IN-priyamvada-medium"),
        ("supply_and_demand_en.json", "Good morning, class! Today we will learn about supply and demand.", "English", "en-US-AriaNeural"),
    ]

    for demo_file, greeting, lang, voice in jobs:
        print(f"Processing {demo_file} ({lang})...")
        try:
            await prepend_greeting(demo_file, greeting, lang, voice)
        except Exception as e:
            print(f"  ERROR on {demo_file}: {e}")
        await asyncio.sleep(1.5)


if __name__ == "__main__":
    asyncio.run(main())
