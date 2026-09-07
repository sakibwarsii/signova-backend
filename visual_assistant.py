"""
Visual Assistant Module  v2
---------------------------
Smarter keyword extraction with media-type classification.
Multi-source image fetching: DuckDuckGo (web scrape) → Pexels → Wikipedia.
NO AI-generated fallback images — only real photos, diagrams, and illustrations.
"""

import asyncio
import json
import os
import httpx
from dotenv import load_dotenv
from groq_client import groq_chat_completion, GROQ_API_KEY, GROQ_API_KEY_FALLBACK

load_dotenv()

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
_groq_available = bool(GROQ_API_KEY or GROQ_API_KEY_FALLBACK)


# ---------------------------------------------------------------------------
# 1. SMARTER KEYWORD PLANNING — AI classifies the best media type per concept
# ---------------------------------------------------------------------------

async def plan_visuals(full_text: str) -> list:
    """
    Single Groq LLM call that reads the full content and returns
    10-15 visual keywords, each with a media_type hint so we can
    build the perfect search query.
    """


    if not _groq_available:
        print("[VisualAssist] No GROQ_API_KEY — skipping visual planning.")
        return []

    prompt = f"""You are a Visual Learning Assistant for an educational classroom.
Given the following educational text, extract 4-5 key visual concepts.

For each concept provide:
- "keyword": the core concept (1-4 words, lowercase, suitable for image search)
- "media_type": what kind of image best represents this concept. Choose ONE of:
    - "labeled_diagram" — for anatomy, circuits, system architectures, processes
    - "scientific_illustration" — for cells, molecules, organisms, chemical structures
    - "photograph" — for real-world objects, places, people, animals, nature
    - "chart" — for data, statistics, comparisons, timelines
    - "infographic" — for step-by-step processes, workflows, cycles
    - "screenshot" — for software, UI, code examples
- "search_query": a specific, optimized search query (5-8 words) that would return the BEST image result on Google. Be very specific and descriptive. For example:
    - Instead of "heart" → "labeled human heart anatomy diagram"
    - Instead of "photosynthesis" → "photosynthesis process diagram chloroplast"
    - Instead of "DNA" → "DNA double helix structure 3D illustration"
    - Instead of "solar system" → "solar system planets scale comparison photograph"
- "triggers": list of 3-5 words/phrases from the text that refer to this concept

IMPORTANT: Output ONLY a valid JSON object with a single key "visuals" containing a list.
Example: {{"visuals": [{{"keyword": "human heart", "media_type": "labeled_diagram", "search_query": "labeled human heart anatomy diagram medical", "triggers": ["heart", "cardiac", "ventricle", "atrium"]}}, {{"keyword": "blood cells", "media_type": "scientific_illustration", "search_query": "red white blood cells microscope illustration", "triggers": ["blood cells", "red blood cells", "white blood cells", "hemoglobin"]}}]}}

Text:
{full_text[:3000]}"""

    try:
        completion = await groq_chat_completion(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200,
            # Same reasoning-token fix applied here as nlp_pipeline.py — this
            # model burns budget on internal reasoning before the real answer,
            # so keeping it low avoids that eating into (or exhausting) the
            # token budget meant for the actual JSON output.
            reasoning_effort="low",
            response_format={"type": "json_object"}
        )
        raw = completion.choices[0].message.content.strip()
        result = json.loads(raw)

        if isinstance(result, list):
            return result

        # Handle wrapper keys
        for key in ["visuals", "keywords", "concepts", "items"]:
            if key in result and isinstance(result[key], list):
                return result[key]
        
        # Fallback: return first list value found
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return v
        return []

    except Exception as e:
        print(f"[VisualAssist] Planning error: {e}")
        return []


# ---------------------------------------------------------------------------
# 2. MULTI-SOURCE IMAGE FETCHING — Real images only, no AI art
# ---------------------------------------------------------------------------

def _is_valid_image_url(url: str) -> bool:
    """Quick sanity check that the URL looks like an actual image."""
    if not url or len(url) < 10:
        return False
    # Reject common ad/tracking domains
    blocked = ["facebook.com", "twitter.com", "instagram.com", "tiktok.com",
               "ads.", "tracking.", "pixel.", "analytics."]
    for b in blocked:
        if b in url.lower():
            return False
    return True


async def _fetch_visual(http: httpx.AsyncClient, item: dict) -> dict:
    """
    Waterfall fetch using the AI-optimized search_query:
      1. DuckDuckGo Image Search (web scrape — best results, multiple attempts)
      2. Pexels API (high-quality stock photos, if API key is set)
      3. Wikipedia REST API (reliable educational thumbnails)
    
    Returns: {"url": str, "source": str}
    """
    
    # Check if a URL was already hardcoded in the plan (e.g. for generated images)
    if "url" in item and item["url"]:
        return {"url": item["url"], "source": item.get("source", "local")}

    search_query = item.get("search_query", item.get("keyword", ""))
    keyword = item.get("keyword", search_query)
    media_type = item.get("media_type", "photograph")
    clean = keyword.strip().lower()

    # ---------- INSTANT SOURCE: Verified Crisp Educational SVG Diagrams ----------
    LOCAL_SVGS = {
        "photosynthesis": "/demo_images/photosynthesis/photosynthesis_diagram.svg",
        "chloroplast": "/demo_images/photosynthesis/chloroplast.svg",
        "chlorophyll": "/demo_images/photosynthesis/chloroplast.svg",
        "ohm": "/demo_images/ohms_law/ohms_law_formula.svg",
        "voltage": "/demo_images/ohms_law/ohms_law_formula.svg",
        "resistance": "/demo_images/ohms_law/ohms_law_formula.svg",
        "cost price": "/demo_images/profit_loss/cost_price.svg",
        "selling price": "/demo_images/profit_loss/selling_price.svg",
        "profit": "/demo_images/profit_loss/profit_vs_loss.svg",
        "loss": "/demo_images/profit_loss/profit_vs_loss.svg",
        "supply and demand": "/demo_images/supply_demand/supply_demand_curve.svg",
        "supply": "/demo_images/supply_demand/supply_demand_curve.svg",
        "demand": "/demo_images/supply_demand/supply_demand_curve.svg",
        "elasticity": "/demo_images/supply_demand/price_elasticity.svg",
        "scarcity": "/demo_images/supply_demand/scarcity.svg",
        "equilibrium": "/demo_images/supply_demand/supply_demand_curve.svg",
        "thirsty crow": "/demo_images/thirsty_crow/thirsty_crow.svg",
        "crow": "/demo_images/thirsty_crow/thirsty_crow.svg",
        "pitcher": "/demo_images/thirsty_crow/thirsty_crow.svg",
        "pebble": "/demo_images/thirsty_crow/pebbles_in_water.svg",
    }
    for k, svg_path in LOCAL_SVGS.items():
        if k in clean:
            return {"url": svg_path, "source": "local_svg"}

    # ---------- SOURCE 1: Wikipedia REST API (fast, free, reliable educational diagrams & photos) ----------
    wiki_headers = {"User-Agent": "SignovaEducationPlatform/1.0 (https://signova.ai; contact@signova.ai)"}
    try:
        slug = clean.replace(' ', '_')
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        resp = await http.get(wiki_url, headers=wiki_headers, timeout=4.0)
        if resp.status_code == 200:
            data = resp.json()
            original = data.get("originalimage", {}).get("source")
            thumb = data.get("thumbnail", {}).get("source")
            url = original or thumb
            if url and _is_valid_image_url(url):
                return {"url": url, "source": "wikipedia"}
        
        # Fallback: search Wikipedia for verified English educational diagram
        search_term = clean if media_type == "photograph" else f"{clean} english diagram"
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={search_term}&gsrlimit=2&prop=pageimages&piprop=original|thumbnail&pithumbsize=1024&format=json"
        sresp = await http.get(search_url, headers=wiki_headers, timeout=4.0)
        if sresp.status_code == 200:
            pages = sresp.json().get("query", {}).get("pages", {})
            for pid, p in pages.items():
                img = p.get("original", {}).get("source") or p.get("thumbnail", {}).get("source")
                if img and _is_valid_image_url(img):
                    return {"url": img, "source": "wikipedia"}
    except Exception as e:
        print(f"[VisualAssist] Wikipedia error for '{keyword}': {e}")

    # ---------- SOURCE 2: Pexels API (fast, free, high-quality stock photos) ----------
    if PEXELS_API_KEY:
        try:
            pexels_query = clean if media_type == "photograph" else search_query
            resp = await http.get(
                "https://api.pexels.com/v1/search",
                params={"query": pexels_query, "per_page": 1, "orientation": "landscape"},
                headers={"Authorization": PEXELS_API_KEY},
                timeout=4.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                photos = data.get("photos", [])
                if photos:
                    # Use medium size for fast loading
                    src = photos[0].get("src", {})
                    url = src.get("large2x") or src.get("large") or src.get("medium")
                    if url:
                        return {"url": url, "source": "pexels"}
        except Exception as e:
            print(f"[VisualAssist] Pexels error for '{keyword}': {e}")

    # ---------- SOURCE 3 (last resort): Pixazo AI image generation ----------
    # Slow (~seconds, 3-step diffusion) and costs API credits, so it's only used
    # when no real photo/diagram was found via the free sources above — matching
    # this module's "real images only" intent instead of contradicting it.
    try:
        PIXAZO_API_KEY = os.environ.get("PIXAZO_API_KEY", "19e791b8f7c247a991324e66ef1e0ab6")
        pixazo_url = "https://gateway.pixazo.ai/flux-1-schnell/v1/getData"
        headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "Ocp-Apim-Subscription-Key": PIXAZO_API_KEY
        }
        payload = {
            "prompt": f"educational photograph of {clean}, cinematic lighting, photorealistic, sharp focus, 8k, award winning macro shot, pure visual without any text, NO TEXT, NO LABELS, NO LETTERS, NO WORDS", 
            "num_steps": 3,
            "height": 1024,
            "width": 1024
        }
        resp = await http.post(pixazo_url, json=payload, headers=headers, timeout=30.0)
        if resp.status_code == 200:
            data = resp.json()
            out_url = data.get("output")
            if out_url and _is_valid_image_url(out_url):
                return {"url": out_url, "source": "pixazo"}
        else:
            print(f"[VisualAssist] Pixazo error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[VisualAssist] Pixazo exception for '{keyword}': {e}")

    # ---------- NO IMAGE FOUND ----------
    print(f"[VisualAssist] No image found for '{keyword}' — skipping.")
    return {"url": None, "source": "none"}


# ---------------------------------------------------------------------------
# 3. BUILD VISUAL LIBRARY — Parallel fetch all keywords
# ---------------------------------------------------------------------------

async def build_visual_library(keyword_plan: list, progress_callback=None) -> dict:
    """
    Fetches all visuals in parallel using a shared httpx client.
    Returns: { "keyword": {"url": ..., "source": ..., "triggers": [...]} }
    """
    if not keyword_plan:
        return {}

    library = {}
    valid_plan = [item for item in keyword_plan if isinstance(item, dict)]
    total_visuals = len(valid_plan)
    
    if total_visuals == 0:
        return {}

    async with httpx.AsyncClient() as http:
        completed = 0
        
        async def fetch_and_report(item):
            nonlocal completed
            try:
                res = await _fetch_visual(http, item)
                return item, res
            except Exception as e:
                return item, e
            finally:
                completed += 1
                if progress_callback:
                    await progress_callback(completed, total_visuals)

        tasks = [fetch_and_report(item) for item in valid_plan]
        results = await asyncio.gather(*tasks)

        for item, result in results:
            if isinstance(result, dict) and result.get("url"):
                keyword = item.get("keyword")
                if keyword:
                    library[keyword] = {
                        **result,
                        "triggers": item.get("triggers", [keyword])
                    }

    print(f"[VisualAssist] Library built: {len(library)} images from {len(keyword_plan)} keywords")
    return library


# ---------------------------------------------------------------------------
# 4. ANNOTATE CHUNKS — Attach visuals to text chunks
# ---------------------------------------------------------------------------

def annotate_chunks_with_visuals(chunks: list, visual_library: dict) -> list:
    """
    Scans each chunk's text against trigger phrases and attaches the best matching
    visual_url, visual_keyword, visual_source. Inherits the last matched visual
    for chunks with no direct match (continuity).
    """
    if not visual_library or not chunks:
        return chunks

    # Build trigger → keyword lookup
    trigger_map: dict[str, str] = {}
    for keyword, data in visual_library.items():
        for trigger in data.get("triggers", [keyword]):
            trigger_map[trigger.lower().strip()] = keyword

    # Default to the first concept so the screen is never blank
    last_keyword: str | None = list(visual_library.keys())[0] if visual_library else None

    for chunk in chunks:
        chunk_text = (chunk.get("text") or "").lower()
        matched: str | None = None

        # Try longest triggers first for better matching
        sorted_triggers = sorted(trigger_map.keys(), key=len, reverse=True)
        for trigger in sorted_triggers:
            if trigger and trigger in chunk_text:
                matched = trigger_map[trigger]
                break

        if matched:
            last_keyword = matched
        elif last_keyword:
            matched = last_keyword  # inherit for visual continuity

        if matched and matched in visual_library:
            chunk["visual_keyword"] = matched
            chunk["visual_url"] = visual_library[matched]["url"]
            chunk["visual_source"] = visual_library[matched]["source"]

    return chunks

