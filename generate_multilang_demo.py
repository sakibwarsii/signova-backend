"""
One-off script: generates a topic's sign/gloss chunks and visuals ONCE, then
produces multiple LANGUAGE VARIANTS (different audio + translated subtitle
text) that all share the exact same sigml and visual_url per chunk — instead
of re-running the whole pipeline (including a fresh, possibly-different
Groq visual-planning call) once per language.

This works because of how the pipeline is actually structured: chunk["text"]
(the English mini-chunk text used to build the sign gloss) is NEVER
overwritten by translation — translate_text_to_language only ever adds a
separate chunk["translated_text"] field. So the sigml and visual_url, both
derived from that unchanging English text, are identical across languages;
only chunk["audio_base64"] (and chunk["translated_text"] for non-English)
differ per language pass.

Usage: edit TOPIC/INSTRUCTIONS/LANGUAGES below, then run.
"""
import asyncio
import copy
import json
import sys

sys.path.insert(0, ".")

from nlp_pipeline import generate_script_from_prompt, nlp, process_text
from visual_assistant import plan_visuals, build_visual_library, annotate_chunks_with_visuals
from ai_tools import get_sigml_for_word_ai, get_mini_chunks, generate_audio_for_chunks


async def build_raw_chunks_from_script(script: str):
    doc = nlp(script)
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


async def build_raw_chunks_from_text(text: str):
    return await build_raw_chunks_from_script(text)


async def generate_topic(topic: str, instructions: str, languages: list, out_prefix: str, from_prompt: bool = True):
    """
    languages: list of (lang, voice, out_suffix) tuples, e.g.
        [("English", "en-US-AriaNeural", "en"), ("Marathi", "mr_IN-google-medium", "mr")]
    """
    if from_prompt:
        print(f"Generating script for topic: {topic}")
        script = await generate_script_from_prompt(topic, instructions)
        # ascii-safe print — the Windows console's default codepage can't print
        # every character the model might use (curly apostrophes, etc.) and
        # crashes the whole script on a plain print(script)
        print("Script:", script.encode("ascii", "replace").decode("ascii"))
    else:
        script = topic  # `topic` param doubles as literal text when from_prompt=False

    raw_chunks = await build_raw_chunks_from_script(script)
    print(f"Built {len(raw_chunks)} sign chunks")

    print("Planning + fetching visuals (once, shared across all languages)...")
    full_text = " ".join(c["text"] for c in raw_chunks)
    keyword_plan = await plan_visuals(full_text)
    visual_library = await build_visual_library(keyword_plan)
    raw_chunks = annotate_chunks_with_visuals(raw_chunks, visual_library)
    with_visuals = sum(1 for c in raw_chunks if c.get("visual_url"))
    print(f"Visuals attached to {with_visuals}/{len(raw_chunks)} chunks")

    for lang, voice, suffix in languages:
        print(f"Generating audio for language: {lang} ({voice})...")
        # Deep copy so each language's audio_base64/translated_text don't clobber the others
        lang_chunks = copy.deepcopy(raw_chunks)
        await generate_audio_for_chunks(lang_chunks, lang, voice, True)
        no_audio = sum(1 for c in lang_chunks if not c.get("audio_base64"))
        out_path = f"F:/speech to sign(updated)/web/public/demo_samples/{out_prefix}_{suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(lang_chunks, f, ensure_ascii=False)
        print(f"  -> wrote {out_path} ({len(lang_chunks)} chunks, {no_audio} missing audio)")
        # Small gap between language passes to stay well clear of free-tier RPM limits
        await asyncio.sleep(3)


async def main():
    # --- Demo 3: The Thirsty Crow — now WITH an English variant too, sharing visuals ---
    crow_text = ("A thirsty crow could not find water. He saw a pitcher with very little water inside. "
                 "He dropped pebbles into the pitcher one by one. The water rose to the top. "
                 "The crow drank the water and flew away happily.")
    await generate_topic(
        topic=crow_text,
        instructions="",
        languages=[
            ("Hindi", "hi_IN-priyamvada-medium", "hi"),
            ("English", "en-US-AriaNeural", "en"),
        ],
        out_prefix="thirsty_crow",
        from_prompt=False,
    )

    print("\n--- cooling off before next topic (stay clear of free-tier RPM) ---\n")
    await asyncio.sleep(8)

    # --- Demo 4: Profit and Loss — Telugu (female) + English ---
    await generate_topic(
        topic="Profit and Loss",
        instructions="Explain simply for young students using a shopkeeper example. Include cost price, selling price, profit, and loss. Keep it under 60 words.",
        languages=[
            ("Telugu", "te_IN-maya-medium", "te"),
            ("English", "en-US-AriaNeural", "en"),
        ],
        out_prefix="profit_and_loss",
        from_prompt=True,
    )

    print("\n--- cooling off before next topic (stay clear of free-tier RPM) ---\n")
    await asyncio.sleep(8)

    # --- Demo 5: Supply and Demand — Malayalam (female) + English ---
    await generate_topic(
        topic="Supply and Demand",
        instructions="Explain simply for young students using an everyday market example. Include price, supply, and demand. Keep it under 60 words.",
        languages=[
            ("Malayalam", "ml_IN-meera-medium", "ml"),
            ("English", "en-US-AriaNeural", "en"),
        ],
        out_prefix="supply_and_demand",
        from_prompt=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
