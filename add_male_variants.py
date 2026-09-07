"""
Adds MALE-voice variants to existing demo topics, reusing each topic's
already-fetched sigml + visual_url (loaded from its English file, which
exists for every topic) instead of re-running script generation and visual
planning — same technique as add_hindi_variant.py.

Marathi is skipped: only one Piper voice exists for it (mr_IN-google-medium,
female) — there's no male Marathi model to switch to yet.
"""
import asyncio
import copy
import json
import sys

sys.path.insert(0, ".")
from ai_tools import generate_audio_for_chunks

DEMO_DIR = "F:/speech to sign(updated)/web/public/demo_samples"


async def add_variant(source_file: str, out_prefix: str, out_suffix: str, lang: str, voice: str):
    with open(f"{DEMO_DIR}/{source_file}", encoding="utf-8") as f:
        source_chunks = json.load(f)

    raw_chunks = []
    for c in source_chunks:
        clean = {"text": c.get("text", ""), "sigml": c.get("sigml", [])}
        for key in ("visual_url", "visual_keyword", "visual_source"):
            if c.get(key):
                clean[key] = c[key]
        raw_chunks.append(clean)

    chunks = copy.deepcopy(raw_chunks)
    await generate_audio_for_chunks(chunks, lang, voice, True)

    no_audio = sum(1 for c in chunks if not c.get("audio_base64"))
    out_path = f"{DEMO_DIR}/{out_prefix}_{out_suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"  -> wrote {out_path} ({len(chunks)} chunks, {no_audio} missing audio)")


async def main():
    jobs = [
        # (source file to reuse visuals/sigml from, out_prefix, out_suffix, lang, male voice)
        ("photosynthesis_en.json", "photosynthesis", "en_male", "English", "en-US-GuyNeural"),

        ("ohms_law_en.json", "ohms_law", "en_male", "English", "en-US-GuyNeural"),
        # Marathi skipped — no male Piper voice exists yet

        ("thirsty_crow_en.json", "thirsty_crow", "en_male", "English", "en-US-GuyNeural"),
        ("thirsty_crow_en.json", "thirsty_crow", "hi_male", "Hindi", "hi_IN-rohan-medium"),

        ("profit_and_loss_en.json", "profit_and_loss", "en_male", "English", "en-US-GuyNeural"),
        ("profit_and_loss_en.json", "profit_and_loss", "hi_male", "Hindi", "hi_IN-rohan-medium"),
        ("profit_and_loss_en.json", "profit_and_loss", "te_male", "Telugu", "te_IN-venkatesh-medium"),

        ("supply_and_demand_en.json", "supply_and_demand", "en_male", "English", "en-US-GuyNeural"),
        ("supply_and_demand_en.json", "supply_and_demand", "hi_male", "Hindi", "hi_IN-rohan-medium"),
        ("supply_and_demand_en.json", "supply_and_demand", "ml_male", "Malayalam", "ml_IN-arjun-medium"),
    ]

    for source_file, out_prefix, out_suffix, lang, voice in jobs:
        print(f"Processing {out_prefix}_{out_suffix} (from {source_file}, {lang})...")
        try:
            await add_variant(source_file, out_prefix, out_suffix, lang, voice)
        except Exception as e:
            print(f"  ERROR on {out_prefix}_{out_suffix}: {e}")
        await asyncio.sleep(1.5)


if __name__ == "__main__":
    asyncio.run(main())
