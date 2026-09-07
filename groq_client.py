"""
Shared Groq client wrapper with automatic fallback to a second account's API
key when the primary is rate-limited or unavailable.

IMPORTANT CAVEAT: Groq's rate limits apply at the ORGANIZATION level, not per
API key. A fallback key only helps if it belongs to a genuinely DIFFERENT
Groq account than the primary — retrying with a second key from the SAME
account would just hit the identical rate-limit bucket again and gain nothing.
GROQ_API_KEY_FALLBACK is expected to be a separate account's key.
"""

import os
from dotenv import load_dotenv
from groq import AsyncGroq, RateLimitError, APIStatusError, APIConnectionError, APITimeoutError

# Loaded here (not left to whichever module happens to import this first) so
# the keys are available regardless of import order — main.py/nlp_pipeline.py/
# visual_assistant.py all import this module, and if one of them imported it
# BEFORE calling its own load_dotenv(), these os.environ.get() calls below
# would read empty strings even though the .env file has real values.
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_KEY_FALLBACK = os.environ.get("GROQ_API_KEY_FALLBACK", "")

if not GROQ_API_KEY:
    print("[WARNING] GROQ_API_KEY not set. AI features will fall back to raw/untranslated output.")
if not GROQ_API_KEY_FALLBACK:
    print("[Groq] No GROQ_API_KEY_FALLBACK set — rate-limit fallback disabled, only the primary key will be used.")

_primary_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
_fallback_client = AsyncGroq(api_key=GROQ_API_KEY_FALLBACK) if GROQ_API_KEY_FALLBACK else None


async def groq_chat_completion(**kwargs):
    """
    Drop-in replacement for `client.chat.completions.create(**kwargs)` that
    automatically retries on the fallback account's key if the primary key:
      - is rate-limited (429 / RateLimitError)
      - is temporarily down / times out (APIConnectionError, APITimeoutError,
        5xx InternalServerError via APIStatusError)
    Errors that are about the REQUEST itself (bad input, auth failure on a
    key that's actually valid, etc.) are NOT retried on the fallback, since a
    different account's key won't fix a malformed request.
    """
    if _primary_client is None and _fallback_client is None:
        raise RuntimeError("No Groq API key configured (GROQ_API_KEY / GROQ_API_KEY_FALLBACK both missing)")

    if _primary_client is None:
        return await _fallback_client.chat.completions.create(**kwargs)

    try:
        return await _primary_client.chat.completions.create(**kwargs)
    except RateLimitError as e:
        print(f"[Groq] Primary key rate-limited: {e}")
    except (APIConnectionError, APITimeoutError) as e:
        print(f"[Groq] Primary key unreachable: {e}")
    except APIStatusError as e:
        # Only retry on server-side failures (5xx) — a 4xx (other than the
        # RateLimitError already caught above) means the request itself is
        # the problem, and the fallback key would fail identically.
        if e.status_code < 500:
            raise
        print(f"[Groq] Primary key returned server error {e.status_code}: {e}")

    # Only reachable if the try block raised one of the caught exceptions
    # above (a successful call already returned from inside the try).
    if _fallback_client is None:
        print("[Groq] No fallback key configured — re-raising original error.")
        raise RuntimeError("Groq primary key failed and no fallback key is configured")

    print("[Groq] Retrying with fallback account's key...")
    return await _fallback_client.chat.completions.create(**kwargs)
