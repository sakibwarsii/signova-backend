"""
Adds a Hindi audio/caption variant to an EXISTING demo topic by reusing its
already-saved sigml + visual_url (loaded straight from one of its other
language files) instead of re-running script generation and visual planning.
Cheaper (skips 2 Groq calls) and guarantees pixel-identical sigml/visuals
across every language variant of the same topic, including this new one.
"""
import asyncio
import copy
import json
import sys

sys.path.insert(0, ".")
from ai_tools import generate_audio_for_chunks

DEMO_DIR = "F:/speech to sign(updated)/web/public/demo_samples"


async def add_hindi(source_file: str, out_prefix: str):
    with open(f"{DEMO_DIR}/{source_file}", encoding="utf-8") as f:
        source_chunks = json.load(f)

    # Strip language-specific fields, keep only what's shared across all
    # variants of this topic: text (used for sigml, untouched by translation),
    # sigml, and visual_url.
    raw_chunks = []
    for c in source_chunks:
        clean = {"text": c.get("text", ""), "sigml": c.get("sigml", [])}
        for key in ("visual_url", "visual_keyword", "visual_source"):
            if c.get(key):
                clean[key] = c[key]
        raw_chunks.append(clean)

    print(f"Loaded {len(raw_chunks)} chunks from {source_file} (visuals reused, not re-fetched)")
    hindi_chunks = copy.deepcopy(raw_chunks)
    await generate_audio_for_chunks(hindi_chunks, "Hindi", "hi_IN-priyamvada-medium", True)

    no_audio = sum(1 for c in hindi_chunks if not c.get("audio_base64"))
    out_path = f"{DEMO_DIR}/{out_prefix}_hi.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(hindi_chunks, f, ensure_ascii=False)
    print(f"  -> wrote {out_path} ({len(hindi_chunks)} chunks, {no_audio} missing audio)")


async def main():
    await add_hindi("profit_and_loss_en.json", "profit_and_loss")
    await asyncio.sleep(3)
    await add_hindi("supply_and_demand_en.json", "supply_and_demand")


if __name__ == "__main__":
    asyncio.run(main())
