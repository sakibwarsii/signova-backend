import asyncio
import copy
import json
import os
import sys
import base64
import tempfile
import edge_tts

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, ".")

from nlp_pipeline import process_text
from ai_tools import get_sigml_for_word_ai

OUT_DIR = r"C:\Users\sakib\.gemini\antigravity\scratch\s2s_web-main\public\demo_samples"
os.makedirs(OUT_DIR, exist_ok=True)

SCRIPTS = [
    # (english_text, visual_url, visual_keyword, translations_dict)
    (
        "Good morning, class!",
        None,
        None,
        {
            "Hindi": "सुप्रभात, कक्षा!",
            "Marathi": "शुभ सकाळ, मुलांनो!",
            "Malayalam": "സുപ്രഭാതം, കൂട്ടുകാരേ!",
            "Telugu": "శుభోదయం, పిల్లలు!",
            "Kannada": "ಶುಭೋದಯ, ಮಕ್ಕಳೇ!"
        }
    ),
    (
        "Today we will explore the",
        None,
        None,
        {
            "Hindi": "आज हम अध्ययन करेंगे",
            "Marathi": "आज आपण माहिती घेणार आहोत",
            "Malayalam": "ഇന്ന് നമ്മൾ പഠിക്കുന്നത്",
            "Telugu": "ఈరోజు మనం తెలుసుకోబోతున్నాం",
            "Kannada": "ಇಂದು ನಾವು ಕಲಿಯೋಣ"
        }
    ),
    (
        "ancient Indus Valley Civilisation.",
        "/demo_images/indus_valley/indus_map.svg",
        "Indus Valley Civilisation",
        {
            "Hindi": "प्राचीन सिंधु घाटी सभ्यता का।",
            "Marathi": "प्राचीन सिंधू संस्कृतीबद्दल.",
            "Malayalam": "പുരാതന സിന്ധുനദീതട സംസ്കാരത്തെക്കുറിച്ചാണ്.",
            "Telugu": "పురాతన సింధు లోయ నాగరికత గురించి.",
            "Kannada": "ಪ್ರಾಚೀನ ಸಿಂಧೂ ಕಣಿವೆ ನಾಗರಿಕತೆಯ ಬಗ್ಗೆ."
        }
    ),
    (
        "It flourished four thousand years",
        "/demo_images/indus_valley/indus_map.svg",
        "Indus River Valley",
        {
            "Hindi": "यह चार हज़ार साल पहले",
            "Marathi": "चार हजार वर्षांपूर्वी",
            "Malayalam": "നാലായിരം വർഷങ്ങൾക്ക് മുൻപ്",
            "Telugu": "ఇది నాలుగు వేల సంవత్సరాల క్రితం",
            "Kannada": "ಇದು ನಾಲ್ಕು ಸಾವಿರ ವರ್ಷಗಳ ಹಿಂದೆ"
        }
    ),
    (
        "ago along the Indus River.",
        "/demo_images/indus_valley/indus_map.svg",
        "River Settlements",
        {
            "Hindi": "सिंधु नदी के किनारे फली-फूली।",
            "Marathi": "सिंधू नदीच्या खोऱ्यात ही संस्कृती बहरली.",
            "Malayalam": "സിന്ധു നദിയുടെ തീരത്താണ് ഇത് വികസിച്ചത്.",
            "Telugu": "సింధు నది తీరంలో వర్ಧిల్లింది.",
            "Kannada": "ಸಿಂಧೂ ನದಿಯ ತೀರದಲ್ಲಿ ಪ್ರವರ್ಧಮಾನಕ್ಕೆ ಬಂದಿತು."
        }
    ),
    (
        "Great cities like Harappa and",
        "/demo_images/indus_valley/great_bath_city_planning.svg",
        "Harappa & Mohenjo-Daro",
        {
            "Hindi": "हड़प्पा और मोहनजोदड़ो जैसे",
            "Marathi": "हडप्पा आणि मोहेंजोदडो सारख्या",
            "Malayalam": "ഹാരപ്പ, മോഹൻജൊദാരോ തുടങ്ങിയ",
            "Telugu": "హరప్పా మరియు మొహంజొదారో వంటి",
            "Kannada": "ಹರಪ್ಪ ಮತ್ತು ಮೊಹೆಂಜೊ-ದಾರೊ ಅಂತಹ"
        }
    ),
    (
        "Mohenjo-daro had brick houses.",
        "/demo_images/indus_valley/great_bath_city_planning.svg",
        "Grid Town Planning",
        {
            "Hindi": "महान नगरों में पक्की ईंटों के मकान थे।",
            "Marathi": "मोठ्या शहरांमध्ये पक्क्या विटांची घरे होती.",
            "Malayalam": "മഹാനഗരങ്ങളിൽ ചുട്ട ഇഷ്ടികകൾ കൊണ്ടുള്ള വീടുകളുണ്ടായിരുന്നു.",
            "Telugu": "గొప్ప నగరాలలో కాల్చిన ఇటుకలతో ఇళ్ళు ఉండేవి.",
            "Kannada": "ದೊಡ್ಡ ನಗರಗಳಲ್ಲಿ ಸುಟ್ಟ ಇಟ್ಟಿಗೆಗಳ ಮನೆಗಳಿದ್ದವು."
        }
    ),
    (
        "They built the famous Great Bath",
        "/demo_images/indus_valley/great_bath_city_planning.svg",
        "The Great Bath",
        {
            "Hindi": "उन्होंने प्रसिद्ध विशाल स्नानागार बनाया",
            "Marathi": "त्यांनी प्रसिद्ध महास्नानगृह बांधले",
            "Malayalam": "അവർ വിഖ്യാതമായ വലിയ കുളിപ്പുരയും",
            "Telugu": "వారు ప్రసిద్ధ మహా స్నానవాటికను నిర్మించారు",
            "Kannada": "ಅವರು ಪ್ರಸಿದ್ಧ ಮಹಾ ಸ್ನಾನಗೃಹವನ್ನು ಕಟ್ಟಿದರು"
        }
    ),
    (
        "and smart covered drainage systems.",
        "/demo_images/indus_valley/great_bath_city_planning.svg",
        "Sanitation & Drainage",
        {
            "Hindi": "और ढकी हुई नालियों की उन्नत व्यवस्था बनाई।",
            "Marathi": "आणि भूमिगत सांडपाण्याची उत्तम व्यवस्था केली.",
            "Malayalam": "അടച്ച ഡ്രെയിനേജ് സംവിധാനങ്ങളും നിർമ്മിച്ചു.",
            "Telugu": "మరియు అధునాతన భూగర్భ మురుగునీటి వ్యవస్థను ఏర్పాటు చేశారు.",
            "Kannada": "ಮತ್ತು ಅತ್ಯುತ್ತಮ ಒಳಚರಂಡಿ ವ್ಯವಸ್ಥೆಯನ್ನು ರೂಪಿಸಿದರು."
        }
    ),
    (
        "Skilled artisans carved stone seals",
        "/demo_images/indus_valley/pashupati_seal_artifacts.svg",
        "Pashupati Stone Seals",
        {
            "Hindi": "कुशल कारीगरों ने पत्थरों की मुहरें तराशीं",
            "Marathi": "कुशल कारागिरांनी सुंदर मुद्रा कोरल्या",
            "Malayalam": "വിദഗ്ദ്ധരായ ശിൽപികൾ കൽമുദ്രകളും",
            "Telugu": "నిపుణులైన శిల్పులు రాతి ముద్రికలను చెక్కారు",
            "Kannada": "ಕುಶಲ ಕರ್ಮಿಗಳು ಸುಂದರವಾದ ಮುದ್ರೆಗಳನ್ನು ಕೆತ್ತಿದರು"
        }
    ),
    (
        "and cast bronze statues.",
        "/demo_images/indus_valley/pashupati_seal_artifacts.svg",
        "Bronze Dancing Girl",
        {
            "Hindi": "और कांस्य की मूर्तियां बनाईं।",
            "Marathi": "आणि कांस्य धातूचे पुतळे बनवले.",
            "Malayalam": "വെങ്കല ശിൽപങ്ങളും നിർമ്മിച്ചു.",
            "Telugu": "మరియు కంచు విగ్రహాలను తయారు చేశారు.",
            "Kannada": "ಮತ್ತು ಕಂಚಿನ ಮೂರ್ತಿಗಳನ್ನು ತಯಾರಿಸಿದರು."
        }
    ),
    (
        "They traded goods and lived",
        "/demo_images/indus_valley/pashupati_seal_artifacts.svg",
        "Maritime Trade & Weights",
        {
            "Hindi": "वे व्यापार करते थे और",
            "Marathi": "ते समृद्ध व्यापार करत असत आणि",
            "Malayalam": "അവർ വ്യാപാരം നടത്തുകയും",
            "Telugu": "వారు వ్యాపారం చేసుకుంటూ",
            "Kannada": "ಅವರು ವ್ಯಾಪಾರ ಮಾಡುತ್ತಾ"
        }
    ),
    (
        "in peaceful planned cities.",
        "/demo_images/indus_valley/pashupati_seal_artifacts.svg",
        "Peaceful Urban Life",
        {
            "Hindi": "योजनाबद्ध शांत नगरों में रहते थे।",
            "Marathi": "नियोजनबद्ध शांत शहरांमध्ये राहत असत.",
            "Malayalam": "സമാധാനപരമായ ആസൂത്രിത നഗരങ്ങളിൽ ജീവിക്കുകയും ചെയ്തു.",
            "Telugu": "ప్రణాళికాబద్ధమైన శాంతియుత నగరాలలో జీవించారు.",
            "Kannada": "ಯೋಜಿತ ಮತ್ತು ಶಾಂತಿಯುತ ನಗರಗಳಲ್ಲಿ ವಾಸಿಸುತ್ತಿದ್ದರು."
        }
    ),
    (
        "Thank you for learning today!",
        None,
        None,
        {
            "Hindi": "आज सीखने के लिए धन्यवाद!",
            "Marathi": "आज अभ्यास केल्याबद्दल धन्यवाद!",
            "Malayalam": "ഇന്ന് പഠിച്ചതിന് എല്ലാവർക്കും നന്ദಿ!",
            "Telugu": "ఈరోజు నేర్చుకున్నందుకు ಧನ್ಯವಾದాలు!",
            "Kannada": "ಇಂದು ಕಲಿತಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು!"
        }
    )
]

LANGUAGES = [
    {
        "name": "English",
        "suffix": "en",
        "female_voice": "en-US-AriaNeural",
        "male_voice": "en-US-GuyNeural"
    },
    {
        "name": "Hindi",
        "suffix": "hi",
        "female_voice": "hi-IN-SwaraNeural",
        "male_voice": "hi-IN-MadhurNeural"
    },
    {
        "name": "Marathi",
        "suffix": "mr",
        "female_voice": "mr-IN-AarohiNeural",
        "male_voice": "mr-IN-ManoharNeural"
    },
    {
        "name": "Malayalam",
        "suffix": "ml",
        "female_voice": "ml-IN-SobhanaNeural",
        "male_voice": "ml-IN-MidhunNeural"
    },
    {
        "name": "Telugu",
        "suffix": "te",
        "female_voice": "te-IN-ShrutiNeural",
        "male_voice": "te-IN-MohanNeural"
    },
    {
        "name": "Kannada",
        "suffix": "kn",
        "female_voice": "kn-IN-SapnaNeural",
        "male_voice": "kn-IN-GaganNeural"
    }
]

async def generate_tts(text: str, voice: str) -> str:
    clean_text = text.replace("<", "").replace(">", "").strip()
    if not clean_text:
        return ""
    communicate = edge_tts.Communicate(clean_text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name
    try:
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        return f"data:audio/mp3;base64,{base64_audio}"
    except Exception as e:
        print(f"TTS error for '{text}' ({voice}): {e}")
        return ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

async def build_base_chunks():
    print("Building base English and SiGML chunks for Indus Valley Civilisation...")
    base_chunks = []
    for en_text, visual_url, visual_keyword, trans_dict in SCRIPTS:
        glosses = process_text(en_text)
        sigml_sequence = []
        for g in glosses:
            sigml_sequence.extend(get_sigml_for_word_ai(g))
        
        chunk = {
            "text": en_text,
            "sigml": sigml_sequence,
            "_translations": trans_dict
        }
        if visual_url:
            chunk["visual_url"] = visual_url
            chunk["visual_keyword"] = visual_keyword
            chunk["visual_source"] = "local_svg"
        base_chunks.append(chunk)
    return base_chunks

async def process_language(base_chunks, lang_spec):
    lang_name = lang_spec["name"]
    suffix = lang_spec["suffix"]
    female_voice = lang_spec["female_voice"]
    male_voice = lang_spec["male_voice"]

    print(f"\n--- Processing {lang_name} ({suffix}) ---")
    
    # Female variant
    female_chunks = []
    for c in base_chunks:
        item = {
            "text": c["text"],
            "sigml": c["sigml"]
        }
        if "visual_url" in c:
            item["visual_url"] = c["visual_url"]
            item["visual_keyword"] = c["visual_keyword"]
            item["visual_source"] = c["visual_source"]
        
        text_to_speak = c["text"]
        if lang_name != "English":
            translated = c["_translations"].get(lang_name, c["text"])
            item["translated_text"] = translated
            text_to_speak = translated
        
        audio = await generate_tts(text_to_speak, female_voice)
        item["audio_base64"] = audio
        female_chunks.append(item)

    out_female = os.path.join(OUT_DIR, f"indus_valley_{suffix}.json")
    with open(out_female, "w", encoding="utf-8") as f:
        json.dump(female_chunks, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Saved female variant: {out_female}")

    # Male variant
    male_chunks = []
    for c, c_fem in zip(base_chunks, female_chunks):
        item = {
            "text": c["text"],
            "sigml": c["sigml"]
        }
        if "visual_url" in c:
            item["visual_url"] = c["visual_url"]
            item["visual_keyword"] = c["visual_keyword"]
            item["visual_source"] = c["visual_source"]
        
        text_to_speak = c["text"]
        if lang_name != "English":
            item["translated_text"] = c_fem.get("translated_text", c["text"])
            text_to_speak = item["translated_text"]
        
        audio = await generate_tts(text_to_speak, male_voice)
        item["audio_base64"] = audio
        male_chunks.append(item)

    out_male = os.path.join(OUT_DIR, f"indus_valley_{suffix}_male.json")
    with open(out_male, "w", encoding="utf-8") as f:
        json.dump(male_chunks, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Saved male variant: {out_male}")

async def main():
    base_chunks = await build_base_chunks()
    print(f"Generated {len(base_chunks)} base chunks.")

    for lang in LANGUAGES:
        await process_language(base_chunks, lang)
        await asyncio.sleep(1)

    print("\nALL 12 INDUS VALLEY DEMO FILES GENERATED SUCCESSFULLY IN ALL 6 LANGUAGES!")

if __name__ == "__main__":
    asyncio.run(main())
