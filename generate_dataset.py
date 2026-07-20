"""
generate_dataset.py

Generates a Q&A fine-tuning dataset for common illnesses (fever, cough,
headache, phlegm, etc.) using Groq's llama-3.1-8b-instant model.

Output: JSONL files in chat-message format, ready for LoRA/QLoRA training
with trl's SFTTrainer or Unsloth.

Setup:
    pip install groq
    export GROQ_API_KEY="your_key_here"
"""

import os
import json
import time
import random
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads .env file in the current directory and loads it into os.environ

# ----------------------------
# CONFIG
# ----------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # set this in your environment
MODEL = "llama-3.1-8b-instant"

OUTPUT_DIR = "dataset"
RAW_FILE = os.path.join(OUTPUT_DIR, "raw_all.jsonl")      # every generated pair, before split
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.jsonl")
VAL_FILE = os.path.join(OUTPUT_DIR, "val.jsonl")

# Topics to generate Q&A pairs for. Add/remove as needed.
TOPICS = [
    "common cold and runny nose",
    "fever in adults",
    "fever in children",
    "headache (tension and mild migraine)",
    "dry cough",
    "wet cough with phlegm/mucus",
    "sore throat",
    "mild stomach ache / indigestion",
    "nausea and mild vomiting",
    "body aches and mild flu symptoms",
    "seasonal allergies",
    "minor cuts and scrapes care",
    "mild dehydration",
    "insomnia / trouble sleeping",
    "constipation",
]

PAIRS_PER_CALL = 4          # smaller batches = fewer tokens per call = fewer rate-limit hits
CALLS_PER_TOPIC = 12        # ~15 topics x 12 calls x 4 pairs = up to 720 raw pairs before dedupe
TRAIN_SPLIT_RATIO = 0.9     # 90% train / 10% val

MAX_TOKENS_PER_CALL = 900   # keep well under the 6000 TPM free-tier limit (prompt + response)
SECONDS_BETWEEN_CALLS = 8   # pacing so you don't hit TPM even before a 429 happens
MAX_RETRIES = 5             # retries on rate-limit errors before giving up on a call

SYSTEM_PROMPT_FOR_FINAL_DATA = (
    "You are a helpful home-health assistant. You give general, safe, "
    "non-diagnostic guidance for common everyday illnesses. You always "
    "recommend seeing a doctor for severe, worsening, or unusual symptoms. "
    "You do not provide specific medication dosages or diagnose conditions."
)

# ----------------------------
# GENERATION PROMPT
# ----------------------------

def build_generation_prompt(topic: str, n_pairs: int, existing_questions: list) -> str:
    avoid_block = ""
    if existing_questions:
        sample = random.sample(existing_questions, min(10, len(existing_questions)))
        avoid_block = (
            "\nDo NOT repeat or closely rephrase these already-used questions:\n"
            + "\n".join(f"- {q}" for q in sample)
        )

    return f"""Generate {n_pairs} realistic question-and-answer pairs about: "{topic}".

Audience: a general person asking about a common, everyday illness at home (not a medical professional).

Rules for the answers:
- General, safe, home-care level advice only (rest, fluids, warm liquids, OTC remedies by category e.g. "a fever reducer" — NOT specific drug names or exact dosages).
- Always mention seeing a doctor if symptoms are severe, persist beyond a few days, or include warning signs (e.g. very high fever, difficulty breathing, blood, severe pain).
- Keep answers practical and easy to understand, 3-6 sentences.
- Vary the phrasing and specific angle of each question (symptoms, causes, home remedies, when to worry, prevention, for children vs adults, etc.) so pairs are not repetitive.
{avoid_block}

Return ONLY a valid JSON array, no markdown, no commentary, in this exact format:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
"""

# ----------------------------
# API CALL
# ----------------------------

def call_groq(client: Groq, prompt: str) -> str:
    """Calls Groq with automatic retry/backoff on rate-limit (429) errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You output only valid JSON arrays. No prose, no markdown fences."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=MAX_TOKENS_PER_CALL,
            )
            return response.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                wait_match = re.search(r"try again in ([\d.]+)s", msg)
                wait_time = float(wait_match.group(1)) + 1 if wait_match else attempt * 10
                print(f"  rate limited, waiting {wait_time:.1f}s (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(wait_time)
                continue
            raise
    raise RuntimeError("Max retries exceeded due to repeated rate limiting.")


def parse_json_array(raw_text: str):
    # Strip markdown fences if the model adds them despite instructions
    cleaned = re.sub(r"```json|```", "", raw_text).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []


# ----------------------------
# MAIN GENERATION LOOP
# ----------------------------

def main():
    if not GROQ_API_KEY:
        raise RuntimeError("Set GROQ_API_KEY as an environment variable before running.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = Groq(api_key=GROQ_API_KEY)

    all_pairs = []
    seen_questions = set()

    # Resume support: if raw_all.jsonl already has data from a previous
    # (possibly interrupted) run, load it so we don't start from scratch
    # and don't regenerate duplicate questions.
    if os.path.exists(RAW_FILE):
        with open(RAW_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    all_pairs.append(item)
                    seen_questions.add(item["question"].lower().strip())
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resuming: loaded {len(all_pairs)} existing pairs from {RAW_FILE}")

    # Open in append mode so each new pair is written to disk immediately,
    # not held in memory until the very end.
    raw_file_handle = open(RAW_FILE, "a", encoding="utf-8")

    for topic in TOPICS:
        print(f"\n=== Topic: {topic} ===")
        for call_num in range(CALLS_PER_TOPIC):
            prompt = build_generation_prompt(topic, PAIRS_PER_CALL, list(seen_questions))
            try:
                raw = call_groq(client, prompt)
                pairs = parse_json_array(raw)
            except Exception as e:
                print(f"  call {call_num+1} failed: {e}")
                continue

            added = 0
            for pair in pairs:
                q = pair.get("question", "").strip()
                a = pair.get("answer", "").strip()
                if not q or not a:
                    continue
                key = q.lower().strip()
                if key in seen_questions:
                    continue
                seen_questions.add(key)
                item = {"topic": topic, "question": q, "answer": a}
                all_pairs.append(item)
                raw_file_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                raw_file_handle.flush()  # force write to disk now, don't buffer
                added += 1

            print(f"  call {call_num+1}: +{added} pairs (total so far: {len(all_pairs)})")
            time.sleep(SECONDS_BETWEEN_CALLS)  # pace calls to stay under free-tier TPM limit

    raw_file_handle.close()
    print(f"\nTotal unique Q&A pairs generated: {len(all_pairs)}")

    # Shuffle and split into train/val
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * TRAIN_SPLIT_RATIO)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    write_chat_format(train_pairs, TRAIN_FILE)
    write_chat_format(val_pairs, VAL_FILE)

    print(f"Saved {len(train_pairs)} -> {TRAIN_FILE}")
    print(f"Saved {len(val_pairs)} -> {VAL_FILE}")


def write_chat_format(pairs: list, filepath: str):
    """Writes pairs in chat-message JSONL format used by SFTTrainer / Unsloth."""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in pairs:
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_FOR_FINAL_DATA},
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
