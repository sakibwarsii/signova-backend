import asyncio
import copy
import json
import os
import sys

sys.path.insert(0, ".")

from nlp_pipeline import translate_text_to_language
from ai_tools import generate_audio_for_chunks

async def main():
    source_file = r"C:\Users\sakib\.gemini\antigravity\scratch\s2s_web-main\public\demo_samples\photosynthesis_en.json"
    
    with open(source_file, "r", encoding="utf-8") as f:
        en_chunks = json.load(f)
        
    print(f"Loaded {len(en_chunks)} chunks. Translating to Kannada...")
    
    # Process for Female Voice (Sapna)
    kn_female_chunks = copy.deepcopy(en_chunks)
    for c in kn_female_chunks:
        if "text" in c and c["text"].strip():
            c["translated_text"] = await translate_text_to_language(c["text"], "Kannada")
        c.pop("audio", None) # Clear English audio
        
    await generate_audio_for_chunks(kn_female_chunks, lang="Kannada", voice="kn-IN-SapnaNeural", tts=True)
    
    out_female = r"C:\Users\sakib\.gemini\antigravity\scratch\s2s_web-main\public\demo_samples\photosynthesis_kn.json"
    with open(out_female, "w", encoding="utf-8") as f:
        json.dump(kn_female_chunks, f)
    print("Saved Female Kannada Demo.")
        
    # Process for Male Voice (Gagan)
    kn_male_chunks = copy.deepcopy(en_chunks)
    for c, c_fem in zip(kn_male_chunks, kn_female_chunks):
        if "translated_text" in c_fem:
            c["translated_text"] = c_fem["translated_text"] # Reuse translation
        c.pop("audio", None)
        
    await generate_audio_for_chunks(kn_male_chunks, lang="Kannada", voice="kn-IN-GaganNeural", tts=True)
    
    out_male = r"C:\Users\sakib\.gemini\antigravity\scratch\s2s_web-main\public\demo_samples\photosynthesis_kn_male.json"
    with open(out_male, "w", encoding="utf-8") as f:
        json.dump(kn_male_chunks, f)
    print("Saved Male Kannada Demo.")

if __name__ == "__main__":
    asyncio.run(main())
