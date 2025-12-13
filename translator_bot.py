# translator_backend.py

import os
import json
import re
import traceback
from typing import List, Optional, Dict

from dotenv import load_dotenv
from openai import OpenAI

from search_hanzi import hvdic_lookup_long


# ==============================
#       LOAD ENV + CLIENT
# ==============================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# Main translation model
MODEL_NAME = "gpt-5.2"

# Model used for extraction (cheap + stable)
MODEL_EXTRACT = "gpt-4o"

# Default translation mode: "smooth" | "literal"
DEFAULT_MODE = "smooth"

# Directory where external prompt files are stored
PROMPT_DIR = "prompts"

# Glossary file
GLOSSARY_PATH = "glossary.json"


# ==============================
#        SMALL UTILITIES
# ==============================

def normalize_key(s: str) -> str:
    """Normalize keys to avoid re-asking due to spaces/newlines."""
    return re.sub(r"\s+", "", str(s or ""))

def strip_code_fence(s: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)

def safe_json_extract_first_object(text: str) -> Optional[dict]:
    """
    Try to parse JSON from text:
    - direct json.loads
    - otherwise extract first {...} block and parse
    """
    t = strip_code_fence(text)
    try:
        return json.loads(t)
    except Exception:
        pass

    first = t.find("{")
    last = t.rfind("}")
    if first != -1 and last != -1 and last > first:
        cand = t[first:last + 1]
        try:
            return json.loads(cand)
        except Exception:
            return None
    return None

def vi_proper_case(s: str) -> str:
    """
    Convert hvdic output (often lowercase / no proper capitalization) into Vietnamese-style Proper Case.
    - Capitalize first letter of each word token.
    - Keep punctuation, hyphens, apostrophes.
    - Avoid messing with all-caps acronyms (rare here).
    """
    s = (s or "").replace("\r", " ").replace("\n", " ").strip()
    if not s:
        return ""

    # Split by spaces but preserve punctuation inside token
    parts = re.split(r"(\s+)", s)

    def cap_token(tok: str) -> str:
        if not tok or tok.isspace():
            return tok

        # If token is purely punctuation
        if re.fullmatch(r"[^\wÀ-ỹ]+", tok, flags=re.UNICODE):
            return tok

        # Handle hyphenated words: "gia-luat" -> "Gia-Luat"
        subparts = tok.split("-")
        out_sub = []
        for sp in subparts:
            if not sp:
                out_sub.append(sp)
                continue

            # Preserve leading/trailing punctuation around word
            m = re.match(r"^([^\wÀ-ỹ]*)([\wÀ-ỹ]+)([^\wÀ-ỹ]*)$", sp, flags=re.UNICODE)
            if not m:
                out_sub.append(sp[:1].upper() + sp[1:])
                continue

            lead, core, tail = m.group(1), m.group(2), m.group(3)
            core2 = core[:1].upper() + core[1:].lower() if len(core) > 1 else core.upper()
            out_sub.append(lead + core2 + tail)

        return "-".join(out_sub)

    return "".join(cap_token(p) for p in parts).strip()


# ==============================
#          PROMPT LOADING
# ==============================

def load_prompt_file(filename: str) -> str:
    path = os.path.join(PROMPT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}. Please create it.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_system_prompt(source_lang: str, mode: str) -> str:
    filename = f"system_{source_lang}_{mode}.txt"
    return load_prompt_file(filename)

def load_assistant_prompt(source_lang: str, mode: str) -> str:
    filename = f"assistant_{source_lang}_{mode}.txt"
    return load_prompt_file(filename)

def load_intro_prompt(source_lang: str) -> str:
    if source_lang == "auto":
        return load_prompt_file("intro_auto.txt")
    filename = f"intro_{source_lang}.txt"
    return load_prompt_file(filename)


# ==============================
#     SPLIT TEXT TO CHUNKS
# ==============================

def split_text_to_chunks(text: str, max_chars: int = 6000) -> List[str]:
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para) if current else para
        else:
            if current.strip():
                chunks.append(current.strip())
            current = para

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ==============================
#   PROPER NOUN EXTRACTION
# ==============================

def extract_proper_nouns(text: str, source_lang: str, max_items: int = 250) -> List[str]:
    """
    Extract proper nouns/titles via LLM → JSON {"items":[...]}.
    Hard rules:
      - Each item must be a short string, single-line.
      - If len(item) > 100 chars => drop.
    Robust parsing + fallback.
    """
    system_prompt = (
        "You are an expert linguistic annotator.\n"
        "Extract ONLY proper nouns and formal titles from the text: person names, place names, "
        "official titles, noble ranks, era names, institutions.\n"
        "Return STRICT JSON only.\n"
        "Schema: {\"items\": [\"...\"]}\n"
        "Rules:\n"
        f"- Return at most {max_items} items.\n"
        "- Each item must be a short single-line string (no newlines).\n"
        "- Do NOT include explanations.\n"
        "- Do NOT include duplicates.\n"
        "- Do NOT include sentences.\n"
        "- No markdown. No code fences.\n"
    )

    user_prompt = f"""
Source language: {source_lang}
Task: Extract unique name/title strings that should be standardized in translation.

Return JSON only:
{{"items": ["..."]}}

Text:
{text}
""".strip()

    resp = client.chat.completions.create(
        model=MODEL_EXTRACT,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = (resp.choices[0].message.content or "").strip()
    raw = strip_code_fence(raw)

    # ---------- helpers ----------
    def clean_item(x: str) -> str:
        x = str(x).replace("\r", " ").replace("\n", " ").strip()
        x = re.sub(r"\s+", " ", x)
        # strip common bullets / numbering
        x = re.sub(r"^(?:[-•*]+)\s*", "", x)
        x = re.sub(r"^\d+[\).\-\s]+", "", x)
        # strip wrapping quotes if any
        if (x.startswith('"') and x.endswith('"')) or (x.startswith("'") and x.endswith("'")):
            x = x[1:-1].strip()
        return x.strip()

    def is_garbage(x: str) -> bool:
        if not x:
            return True
        if len(x) > 100:          # <-- đúng yêu cầu của bạn
            return True
        # loại các dòng kiểu JSON fragments / meta
        if any(ch in x for ch in ["{", "}", "[", "]"]):
            return True
        if x.lower().startswith(("items:", "json", "output", "schema")):
            return True
        # loại các dòng quá “câu văn” (nhiều dấu câu)
        if len(re.findall(r"[，。；;,:]", x)) >= 3:
            return True
        return False

    def dedupe_strong(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for it in items:
            it2 = clean_item(it)
            if is_garbage(it2):
                continue
            key = normalize_key(it2)
            if key and key not in seen:
                seen.add(key)
                out.append(it2)
            if len(out) >= max_items:
                break
        return out

    # ---------- parse layer 1: strict json ----------
    data = safe_json_extract_first_object(raw)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return dedupe_strong(data["items"])

    # ---------- parse layer 2: try to find "items": [...] and parse that list ----------
    m = re.search(r'"items"\s*:\s*\[(.*?)\]', raw, flags=re.DOTALL)
    if m:
        cand = '{"items":[' + m.group(1) + ']}'
        data2 = safe_json_extract_first_object(cand)
        if isinstance(data2, dict) and isinstance(data2.get("items"), list):
            return dedupe_strong(data2["items"])

    # ---------- fallback: split raw by separators (last resort) ----------
    rough = re.split(r'[\n,，、;；]+', raw)
    return dedupe_strong(rough)

# ==============================
#        GLOSSARY I/O
# ==============================

def load_glossary(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    # Ensure all values are strings
    return {str(k): str(v) for k, v in data.items()}

def save_glossary(path: str, glossary: Dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)


# ==============================
#   BUILD GLOSSARY FROM HVDIC
# ==============================

def build_glossary_from_hvdic(
    names: List[str],
    *,
    existing: Optional[Dict[str, str]] = None,
    print_result: bool = True,
    max_len: int = 80,
) -> Dict[str, str]:
    """
    Auto-build mapping using hvdic_lookup_long.
    - Only lookup each missing item once (caller passes missing list; this also double-checks existing).
    - Print lookup results to console so you can verify.
    - Proper-case output (Triệu Khuông Dận, Yên Kinh, Trung thư thị lang...)
    - Skip too-long strings to avoid tool errors.
    """
    out: Dict[str, str] = {}
    existing = existing or {}

    total = len(names)
    if total == 0:
        return out

    print(f"📚 Tra Hán–Việt bằng hvdic cho {total} mục (mỗi mục 1 lần)...")

    for i, n in enumerate(names, 1):
        key = str(n).replace("\r", " ").replace("\n", " ").strip()
        if not key:
            continue

        # Nếu đã có trong glossary thì bỏ qua (đảm bảo không tra lại)
        if key in existing or key in out:
            if print_result:
                print(f"⏭️  [{i}/{total}] (đã có) {key} -> {existing.get(key) or out.get(key)}")
            continue

        # Chặn các cụm "quá dài" (thường là do extract nhầm, không phải tên riêng)
        if len(key) > max_len:
            if print_result:
                print(f"⚠️  [{i}/{total}] BỎ QUA (quá dài {len(key)}): {key[:50]}...")
            continue

        try:
            hv_raw = hvdic_lookup_long(key)
        except Exception as e:
            hv_raw = None

        hv = vi_proper_case(hv_raw) if hv_raw else ""

        # Lọc nhiễu phổ biến của tool (phòng trường hợp tool trả về trang lỗi)
        bad_markers = ["Lightgoldenrodyellow", "Viewport", "Quá Giới Hạn", "timchu", "phienam"]
        if any(m.lower() in hv.lower() for m in bad_markers):
            hv = ""

        if hv:
            out[key] = hv
            if print_result:
                print(f"🔎 [{i}/{total}] {key} -> {hv}")
        else:
            if print_result:
                print(f"❌ [{i}/{total}] {key} -> (không ra)")

    print(f"✅ Tra xong: lấy được {len(out)}/{total} mục.\n")
    return out



def build_glossary_interactively(names: List[str]) -> Dict[str, str]:
    """
    Optional manual fill for unresolved items.
    """
    glossary = {}
    if not names:
        return glossary

    print("\n=== Manual Glossary (fallback) ===")
    print("Nhập cách viết bạn muốn (Enter để bỏ qua).")
    for i, n in enumerate(names, 1):
        val = input(f"[{i}/{len(names)}] {n} => ").strip()
        if val:
            glossary[n] = val
    return glossary


# ==============================
#      TRANSLATION CALL
# ==============================

def translate_chunk(chunk: str, mode: str, source_lang: str, glossary: Optional[Dict[str, str]] = None) -> str:
    system_prompt = load_system_prompt(source_lang, mode)
    assistant_prompt = load_assistant_prompt(source_lang, mode)
    intro_text = load_intro_prompt(source_lang)

    glossary_text = ""
    if glossary:
        glossary_text = (
            "=== GLOSSARY (THAM KHẢO, HOA/THƯỜNG tùy theo hoàn cảnh mà sửa đổi) ===\n"
        )
        for k, v in glossary.items():
            glossary_text += f"- {k} => {v}\n"
        glossary_text += "=== END GLOSSARY ===\n\n"

    user_content = intro_text + "\n\n" + glossary_text + chunk

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


def translate_big_text(
    text: str,
    mode: str = DEFAULT_MODE,
    source_lang: str = "zh",
    glossary: Optional[Dict[str, str]] = None,
) -> str:
    text = text.strip()
    if not text:
        return ""

    if len(text) <= 6000:
        return translate_chunk(text, mode=mode, source_lang=source_lang, glossary=glossary)

    chunks = split_text_to_chunks(text, max_chars=6000)
    print(f"🔍 Long text detected → split into {len(chunks)} chunks.\n")

    translated_chunks: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        print(f"⏳ Translating chunk {i}/{len(chunks)} ...")
        t = translate_chunk(ch, mode=mode, source_lang=source_lang, glossary=glossary)
        translated_chunks.append(t)
        print(f"   ✔ Done chunk {i}\n")

    return "\n\n".join(translated_chunks)


# ==============================
#              MAIN
# ==============================

def main():
    print("=== Translation CLI ===")
    print("This script will translate a text file into Vietnamese.")
    print("Supported language codes:")
    print("  auto, zh, hv, en, th, lo, fr, other")
    print()

    input_path = input("Enter input file path (default: input.txt): ").strip() or "input.txt"
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return

    output_path = input("Enter output file path (default: output.txt): ").strip() or "output.txt"

    source_lang = input(
        "Enter source language code [auto, zh, hv, en, th, lo, fr, other] (default: zh): "
    ).strip().lower() or "zh"

    mode = DEFAULT_MODE

    with open(input_path, "r", encoding="utf-8") as f:
        sample_text = f.read().strip()

    if not sample_text:
        print("❌ Input file is empty, nothing to translate.")
        return

    # ===== Glossary step (AUTO: hvdic_lookup_long) =====
    use_glossary = input("Bạn có muốn tự động chuẩn hóa tên riêng bằng hvdic? (Y/n): ").strip().lower()
    use_glossary = (use_glossary != "n")

    glossary: Dict[str, str] = {}
    if use_glossary:
        if os.path.exists(GLOSSARY_PATH):
            reuse = input("Đã có glossary.json. Dùng lại và bổ sung? (Y/n): ").strip().lower()
            if reuse != "n":
                glossary = load_glossary(GLOSSARY_PATH)

        # normalize map for "already have" check
        norm_existing = {normalize_key(k): k for k in glossary.keys()}

        # Extract names (avoid too-long extraction)
        print("🔎 Đang trích xuất tên riêng/chức tước từ văn bản...")
        extract_text = sample_text[:20000]  # giới hạn để ổn định
        names = extract_proper_nouns(extract_text, source_lang=source_lang)

        # Determine missing (use normalize to avoid re-asking)
        missing = []
        for n in names:
            nk = normalize_key(n)
            if nk and nk not in norm_existing:
                missing.append(n)

        print(f"📌 Extracted {len(names)} mục, còn thiếu {len(missing)} mục so với glossary hiện có.")

        if missing:
            print("📚 Đang tra hvdic_lookup_long cho các mục còn thiếu...")
            auto_map = build_glossary_from_hvdic(missing, existing=glossary, print_result=True)
            glossary.update(auto_map)
            print(f"✅ Tra xong: bổ sung được {len(auto_map)} mục từ hvdic.\n")

            # optional: manual fill for ones still missing
            still_missing = [n for n in missing if normalize_key(n) not in {normalize_key(k) for k in glossary.keys()}]
            if still_missing:
                ask_more = input(f"Còn {len(still_missing)} mục hvdic không ra. Muốn nhập tay? (y/N): ").strip().lower() == "y"
                if ask_more:
                    manual_map = build_glossary_interactively(still_missing)
                    # also proper-case manual if you want: leave as user typed (safer)
                    glossary.update(manual_map)

        save_glossary(GLOSSARY_PATH, glossary)
        print(f"💾 Đã lưu glossary vào {GLOSSARY_PATH}\n")

    print("⏳ Translating...")

    input("\n=== Press Enter to continue ===\n")

    glossary = load_glossary(GLOSSARY_PATH)
    translated = translate_big_text(
        sample_text,
        mode=mode,
        source_lang=source_lang,
        glossary=glossary if glossary else None,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated)

    print(f"✅ Done. Result saved to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("==== FATAL ERROR ====")
        print("Error:", e)
        traceback.print_exc()
        raise
