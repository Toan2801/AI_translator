# translator_backend.py

import traceback
import os
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# ==============================
#       LOAD ENV + CLIENT
# ==============================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    # base_url="https://api.deepseek.com"
)

MODEL_NAME = "gpt-5.1"

# Default translation mode: "smooth" | "literal"
DEFAULT_MODE = "literal"

# Directory where external prompt files are stored
PROMPT_DIR = "prompts"

# ==============================
#   LOAD EXAMPLE CONTEXT (STYLE)
# ==============================

EXAMPLE_CONTEXT: Optional[str] = None
EXAMPLE_PATH = "example.txt"

if os.path.exists(EXAMPLE_PATH):
    try:
        with open(EXAMPLE_PATH, "r", encoding="utf-8") as f:
            EXAMPLE_CONTEXT = f.read().strip()
        print("📘 Loaded example.txt as style reference.")
    except Exception as e:
        print(f"⚠ Cannot read example.txt: {e}")
        EXAMPLE_CONTEXT = None


# ==============================
#          PROMPT LOADING
# ==============================

def load_prompt_file(filename: str) -> str:
    """
    Load a prompt text file from PROMPT_DIR.
    If the file does not exist or cannot be read, raise a clear error.
    """
    path = os.path.join(PROMPT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prompt file not found: {path}. Please create it."
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Cannot read prompt file {path}: {e}")



def load_system_prompt(source_lang: str, mode: str) -> str:
    """
    Load the system prompt text from a .txt file.

    File naming convention:
        prompts/system_{lang_key}_{mode}.txt

    Examples:
        system_zh_smooth.txt
        system_zh_literal.txt
        system_hv_smooth.txt
        system_general_smooth.txt
    """
    filename = f"system_{source_lang}_{mode}.txt"
    return load_prompt_file(filename)


def load_assistant_prompt(source_lang: str, mode: str) -> str:
    """
    Load the assistant guidelines prompt from a .txt file.

    File naming convention:
        prompts/assistant_{lang_key}_{mode}.txt

    Examples:
        assistant_zh_smooth.txt
        assistant_hv_literal.txt
        assistant_general_smooth.txt
    """
    filename = f"assistant_{source_lang}_{mode}.txt"
    return load_prompt_file(filename)


def load_intro_prompt(source_lang: str) -> str:
    """
    Load the intro part that will be prepended to the user content.
    This is the small instruction text like:
        - 'This is Hán-Việt phonetic, read as ...'
        - 'The following text should be translated to Vietnamese...'
        - 'Detect language automatically...'

    File naming convention:
        prompts/intro_{lang_key}.txt

    Examples:
        intro_zh.txt
        intro_hv.txt
        intro_general.txt
        intro_auto.txt   (optional if you want a different intro for auto mode)
    """
    # For auto detection, you might want a dedicated intro file.
    if source_lang == "auto":
        filename = "intro_auto.txt"
        return load_prompt_file(filename)

    filename = f"intro_{source_lang}.txt"
    return load_prompt_file(filename)


# ==============================
#     SPLIT TEXT TO CHUNKS
# ==============================

def split_text_to_chunks(text: str, max_chars: int = 6000) -> List[str]:
    """
    Split text into multiple chunks by paragraphs (separated by two newlines),
    each chunk <= max_chars characters, suitable for sending to the model.
    """
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        # +2 accounts for the two newlines we add when concatenating
        if len(current) + len(para) + 2 <= max_chars:
            if current:
                current += "\n\n" + para
            else:
                current = para
        else:
            if current.strip():
                chunks.append(current.strip())
            current = para

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ==============================
#      TRANSLATION CALL
# ==============================

def translate_chunk(chunk: str, mode: str, source_lang: str) -> str:
    """
    Translate a single chunk of text.
    - Loads system prompt, assistant guidelines, and intro text from .txt files.
    - Optionally prepends EXAMPLE_CONTEXT as style reference.
    """
    system_prompt = load_system_prompt(source_lang, mode)
    assistant_prompt = load_assistant_prompt(source_lang, mode)
    intro_text = load_intro_prompt(source_lang)

    if EXAMPLE_CONTEXT:
        user_content = (
            "Dưới đây là MẪU DỊCH tiếng Việt (chỉ dùng để tham khảo phong cách, KHÔNG dịch lại phần này):\n\n"
            f"{EXAMPLE_CONTEXT}\n\n"
            "-----\n"
            + intro_text
            + "\n\n"
            + chunk
        )
    else:
        user_content = intro_text + "\n\n" + chunk

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": assistant_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("==== ERROR WHEN CALLING API ====")
        print("Error:", e)
        traceback.print_exc()
        raise


def translate_big_text(
    text: str,
    mode: str = DEFAULT_MODE,
    source_lang: str = "hv",
) -> str:
    """
    Main backend function to translate an arbitrary-length text.

    Args:
        text: input text to translate.
        mode: "literal" | "smooth".
        source_lang: "auto" | "zh" | "hv" | "th" | "lo" | "en" | "fr" | "other".
                 Mỗi mã sẽ tìm các file:
                 system_{source_lang}_{mode}.txt,
                 assistant_{source_lang}_{mode}.txt,
                 intro_{source_lang}.txt
                 (riêng auto dùng intro_auto.txt)
    """
    text = text.strip()
    if not text:
        return ""

    if len(text) <= 6000:
        return translate_chunk(text, mode=mode, source_lang=source_lang)

    chunks = split_text_to_chunks(text, max_chars=6000)
    print(f"🔍 Long text detected → split into {len(chunks)} chunks.\n")

    translated_chunks: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        print(f"⏳ Translating chunk {i}/{len(chunks)} ...")
        t = translate_chunk(ch, mode=mode, source_lang=source_lang)
        translated_chunks.append(t)
        print(f"   ✔ Done chunk {i}\n")

    return "\n\n".join(translated_chunks)


# ==============================
#              MAIN
# ==============================

def main():
    """
    CLI entry point.

    Flow:
      1. Ask for input file path
      2. Ask for output file path
      3. Ask for source language code
    """
    print("=== Translation CLI ===")
    print("This script will translate a text file into Vietnamese.")
    print("Supported language codes:")
    print("  auto, zh, hv, en, th, lo, fr, other")
    print()

    # 1) Ask for input file path
    input_path = input("Enter input file path (default: input.txt): ").strip()
    if not input_path:
        input_path = "input.txt"

    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return

    # 2) Ask for output file path
    output_path = input("Enter output file path (default: output.txt): ").strip()
    if not output_path:
        output_path = "output.txt"

    # 3) Ask for source language code
    source_lang = input(
        "Enter source language code [auto, zh, hv, en, th, lo, fr, other] (default: zh): "
    ).strip().lower()
    if not source_lang:
        source_lang = "zh"

    # Optional: if you want to choose mode interactively, uncomment this:
    # mode_in = input("Enter mode [literal/smooth] (default: smooth): ").strip().lower()
    # mode = mode_in if mode_in in ("literal", "smooth") else DEFAULT_MODE

    mode = DEFAULT_MODE

    # Read input text
    with open(input_path, "r", encoding="utf-8") as f:
        sample_text = f.read().strip()

    if not sample_text:
        print("❌ Input file is empty, nothing to translate.")
        return

    print("⏳ Translating...")

    translated = translate_big_text(
        sample_text,
        mode=mode,
        source_lang=source_lang,
    )

    # Write output text
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated)

    print(f"✅ Done. Result saved to {output_path}")


if __name__ == "__main__":
    main()
