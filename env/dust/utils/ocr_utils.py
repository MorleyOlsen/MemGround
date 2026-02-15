import os
import json
import base64
import ssl
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from typing import Dict, Any, List, Optional, Tuple

REQUEST_URL = "https://gjbsb.market.alicloudapi.com/ocrservice/advanced"
context = ssl._create_unverified_context()

def get_img_payload(img_file: str) -> Tuple[str, str]:
    """返回 ('url', url) 或 ('img', base64)"""
    if img_file.startswith("http://") or img_file.startswith("https://"):
        return ("url", img_file)

    path = os.path.expanduser(img_file)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return ("img", b64)

def posturl(headers: Dict[str, str], body: Dict[str, Any]) -> str:
    """发送请求，返回 response 文本"""
    try:
        params = json.dumps(body).encode("utf-8")
        req = Request(REQUEST_URL, params, headers)
        r = urlopen(req, context=context, timeout=30)
        html = r.read()
        return html.decode("utf-8")
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTPError {e.code}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"URLError: {e}") from e

def parse_aliyun_ocr_words(resp_text: str) -> str:
    data = json.loads(resp_text)
    words_info = data.get("prism_wordsInfo", [])
    words = []
    if isinstance(words_info, list):
        for item in words_info:
            w = item.get("word")
            if isinstance(w, str) and w.strip():
                words.append(w.strip())
    return "\n".join(words)

def request_one(appcode: str, img_file: str, extra_params: Optional[Dict[str, Any]] = None) -> str:
    """识别单张图片，返回拼好的文本"""
    params: Dict[str, Any] = {}
    if extra_params:
        params.update(extra_params)

    key, val = get_img_payload(img_file)
    params[key] = val

    headers = {
        "Authorization": f"APPCODE {appcode}",
        "Content-Type": "application/json; charset=UTF-8",
    }

    resp_text = posturl(headers, params)
    return parse_aliyun_ocr_words(resp_text)

def request_many(appcode: str, img_files: List[str], extra_params: Optional[Dict[str, Any]] = None) -> str:
    """识别图片列表，把所有识别结果拼成一段"""
    parts: List[str] = []
    for i, img in enumerate(img_files, 1):
        try:
            text = request_one(appcode, img, extra_params)
            # 下面这行决定“每张图之间怎么拼”
            # 如果你要完全连续一段：parts.append(text)
            parts.append(text)
        except Exception as e:
            parts.append(f"【第{i}张】\n<OCR FAILED: {e}>")

    return "\n\n".join(parts)

# 示例
if __name__ == "__main__":
    APPCODE = os.environ.get("ALI_OCR_APPCODE", "").strip()
    imgs = ["./a.png", "./b.jpg"]
    merged = request_many(APPCODE, imgs)
    print(merged)
