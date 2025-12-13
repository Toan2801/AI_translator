import time
import requests
import re
from typing import List
import html as html_lib

# URL PHP phiên âm Hán-Việt
NGUYENDU_URL = "http://nguyendu.com.free.fr/hanviet/hv_phienam_dtk.php"


def _call_nguyendu(text: str) -> str:
    """
    Gửi MỘT ĐOẠN chữ Hán lên hv_phienam_dtk.php, trả về HTML kết quả.
    """
    data = {
        "choix": "2",          # 2 cột
        "choix_py": "4",       # Hán Việt
        "nbhanchar": "1000",   # mỗi đoạn <= 1000 Hán tự
        "text_in": text,       # nội dung chữ Hán cần phiên âm
        "submit": "Go",
    }

    resp = requests.post(
        NGUYENDU_URL,
        data=data,
        timeout=2,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    # đảm bảo decode UTF-8
    resp.encoding = "utf-8"
    resp.raise_for_status()
    return resp.text


def _parse_phienam_from_html(html: str) -> str:
    """
    Rút ra các khối phiên âm Hán-Việt từ HTML.

    Trang hv_phienam_dtk.php hiển thị phiên âm trong:
        <div class='div-td-scrolls'> ... </div>

    Bên trong div đó là mớ <FONT>...</FONT> chứa từng âm,
    ta chỉ việc:
      - html.unescape
      - xóa tag
      - nén khoảng trắng.
    """
    # Tìm tất cả các div có class = div-td-scrolls
    blocks = re.findall(
        r"<div[^>]+class=['\"]div-td-scrolls['\"][^>]*>(.*?)</div>",
        html,
        flags=re.S | re.I,
    )

    if blocks:
        parts = []
        for b in blocks:
            # Giải mã entity như &#65292; -> ，
            seg = html_lib.unescape(b)
            # Xóa hết tag HTML bên trong (FONT, BR, v.v.)
            seg = re.sub(r"<[^>]+>", "", seg)
            # Nén khoảng trắng / loại bỏ xuống dòng nội bộ
            seg = re.sub(r"\s+", " ", seg).strip()
            if seg:
                parts.append(seg)

        # Ghép các khối thành một chuỗi (nếu có nhiều khối cho cùng một input)
        return " ".join(parts)

    # Fallback: nếu không tìm được div-td-scrolls (trang đổi giao diện)
    m_body = re.search(r"<body[^>]*>(.*?)</body>", html, flags=re.S | re.I)
    body = m_body.group(1) if m_body else html
    body = re.sub(r"<script.*?</script>", "", body, flags=re.S | re.I)
    body = re.sub(r"<style.*?</style>", "", body, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", "", body)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_phienam_text(s: str) -> str:
    """
    Làm sạch text phiên âm:
    - Bỏ xuống dòng / tab nội bộ → đổi thành khoảng trắng.
    - Nén khoảng trắng.
    - Bỏ cụm ngoặc Hán đầu dòng: 【宋紀六十】 ...
    - Bỏ Hán tự đơn lẻ ở đầu dòng: 仁 nhân..., 九 cửu...
    - Khử một số duplicate như '【【' → '【', '】】' → '】'.
    """
    # Bỏ các xuống dòng nội bộ (giữa cùng một dòng logic)
    s = s.replace("\r", " ").replace("\n", " ")
    # Nén khoảng trắng
    s = re.sub(r"\s+", " ", s).strip()

    # Khử trường hợp hai dấu ngoặc liền nhau
    s = re.sub(r"【\s*【", "【", s)
    s = re.sub(r"】\s*】", "】", s)


    # Nếu đầu dòng là 1 hoặc vài chữ Hán rồi tới khoảng trắng: 仁 nhân..., 九 cửu...
    #    → bỏ phần Hán ở đầu, giữ lại phiên âm
    s = re.sub(r"^[\u4E00-\u9FFF]+(\s+)", "", s)

    return s.strip()



def _split_by_length(text: str, max_chars: int = 1000) -> List[str]:
    """
    Cắt text thành các đoạn tối đa max_chars ký tự.
    (không đụng đến xuống dòng, hàm này dùng cho 1 dòng đơn).
    """
    chunks: List[str] = []
    n = len(text)
    for i in range(0, n, max_chars):
        chunks.append(text[i:i + max_chars])
    return chunks


def _lookup_line(line: str, max_chars: int = 1000, sleep_sec: float = 0.0) -> str:
    """
    Phiên âm Hán-Việt CHO MỘT DÒNG duy nhất.

    - Nếu dòng ngắn <= max_chars: gọi PHP 1 lần.
    - Nếu dòng dài: cắt thành nhiều đoạn, gọi nhiều lần rồi ghép lại.
    - Luôn trả về 1 dòng Hán-Việt (không chèn \n).
    """
    line = line.rstrip("\r\n")
    if not line.strip():
        return ""

    # Nếu đủ ngắn, gọi 1 lần
    if len(line) <= max_chars:
        try:
            html = _call_nguyendu(line)
            phienam = _parse_phienam_from_html(html)
            phienam = _clean_phienam_text(phienam)
            return phienam
        except Exception as e:
            print(f"Lỗi khi gọi nguyendu cho dòng: {e}. Giữ nguyên dòng gốc.")
            return line

    # Nếu quá dài, cắt nhỏ hơn
    chunks = _split_by_length(line, max_chars=max_chars)
    out_parts: List[str] = []

    for idx, chunk in enumerate(chunks, 1):
        if not chunk.strip():
            continue
        try:
            html = _call_nguyendu(chunk)
            phienam = _parse_phienam_from_html(html)
            phienam = _clean_phienam_text(phienam)
            out_parts.append(phienam)
            print(f"  Đã xử lý đoạn {idx}/{len(chunks)} của một dòng (độ dài {len(chunk)} ký tự).")
        except Exception as e:
            print(f"Lỗi khi gọi nguyendu cho đoạn {idx} trong một dòng: {e}. Giữ nguyên đoạn gốc.")
            out_parts.append(chunk)

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    # Ghép các phần lại thành 1 dòng duy nhất
    return " ".join(out_parts).strip()


def hvdic_lookup_long(text: str, max_chars: int = 1000, sleep_sec: float = 0.0) -> str:
    """
    Tra Hán-Việt cho CẢ ĐOẠN VĂN dài bằng hv_phienam_dtk.php.

    YÊU CẦU (đã đảm bảo):
    - Chỗ nào input xuống dòng → output xuống dòng y như vậy.
    - Không tự tiện thêm/bớt dòng trống.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.rstrip()
    if not text:
        return ""

    input_lines = text.split("\n")
    output_lines: List[str] = []

    for line_idx, line in enumerate(input_lines, 1):
        # Giữ nguyên dòng trống
        if not line.strip():
            output_lines.append("")
            continue

        out_line = _lookup_line(line, max_chars=max_chars, sleep_sec=sleep_sec)
        output_lines.append(out_line)

    # Ghép lại với đúng số dòng như input
    return "\n".join(output_lines)


def process_file(input_path: str = "input.txt", output_path: str = "output.txt"):
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Lỗi đọc {input_path}: {e}")
        return

    print("⏳ Đang tra Hán-Việt qua hv_phienam_dtk.php ...")

    hv_text = hvdic_lookup_long(text)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(hv_text)
        print(f"✅ Hoàn tất! Đã ghi kết quả vào {output_path}")
    except Exception as e:
        print(f"Lỗi ghi {output_path}: {e}")


if __name__ == "__main__":
    # 1) Ask for input file path
    input_path = input("Enter input file path (default: input.txt): ").strip()
    if not input_path:
        input_path = "input.txt"

    # 2) Ask for output file path
    output_path = input("Enter output file path (default: output.txt): ").strip()
    if not output_path:
        output_path = "output.txt"
    process_file(input_path, output_path)