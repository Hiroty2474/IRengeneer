# ocr_llm_proofread.py  — 画像ファイル選択ダイアログ対応版
import os, sys, json, time, argparse, re, glob, textwrap, traceback
from typing import List, Dict, Any

# ====== 0) 環境パス（ASCII固定） ======
BASE = r"C:\Hiroto\IR"
HF_HOME  = os.path.join(BASE, ".hf")
OV_CACHE = os.path.join(BASE, ".ovcache")
TMP_DIR  = os.path.join(BASE, ".tmp")
for p in (HF_HOME, OV_CACHE, TMP_DIR):
    os.makedirs(p, exist_ok=True)
os.environ.setdefault("TEMP", TMP_DIR)
os.environ.setdefault("TMP", TMP_DIR)
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "tf"))
os.environ.setdefault("OPENVINO_CACHE_DIR", OV_CACHE)

# ====== 1) 依存（LLM / EasyOCR） ======
import openvino as ov
import openvino_genai as ov_genai

def _load_ocr():
    import easyocr
    # 日英混在に強くしたいなら ['ja','en'] に
    langs = ['ja']
    return ("easyocr", easyocr.Reader(langs))

def run_ocr(backend: str, ocr, image_path: str) -> dict:
    # --- Unicodeパス安全読み込み ---
    import numpy as np, cv2, shutil
    img = None
    try:
        with open(image_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img = None

    if img is None:
        # フォールバック: ASCII一時フォルダにコピーしてパス入力で読む
        tmp_dir = os.environ.get("TMP", r"C:\Hiroto\IR\.tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, "ocr_work.png")
        try:
            shutil.copy2(image_path, tmp_path)
            res = ocr.readtext(tmp_path, detail=1, paragraph=False)
        except Exception as e:
            raise RuntimeError(f"画像を開けませんでした: {image_path} ({e})")
    else:
        # ndarrayを直接渡す（これが最も確実）
        res = ocr.readtext(img, detail=1, paragraph=False)

    lines = []
    for (box, txt, conf) in res:
        if str(txt).strip():
            lines.append({"text": str(txt), "conf": float(conf), "box": box})
    joined = "\n".join([l["text"] for l in lines])
    return {"lines": lines, "text": joined}

def load_llm(model_repo_or_dir: str, device: str, max_prompt_len=512, min_resp_len=16):
    print(f"[LLM] init device={device} model={model_repo_or_dir}")
    pipe = ov_genai.LLMPipeline(
        model_repo_or_dir,
        device,
        CACHE_DIR=OV_CACHE,
        GENERATE_HINT="BEST_PERF",
        MAX_PROMPT_LEN=max_prompt_len,
        MIN_RESPONSE_LEN=min_resp_len,
    )
    return pipe

# ====== 2) ダイアログ（複数選択OK） ======
def pick_images_with_dialog(last_dir_hint: str | None = None) -> list[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return []  # GUIが使えない環境

    root = tk.Tk()
    root.withdraw()
    options = {
        "title": "OCRする画像を選択（複数可）",
        "filetypes": [
            ("画像ファイル", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
            ("すべて", "*.*"),
        ],
        "multiple": True,
    }
    if last_dir_hint and os.path.isdir(last_dir_hint):
        options["initialdir"] = last_dir_hint
    paths = filedialog.askopenfilenames(**options)
    root.destroy()
    return list(paths)

def load_last_dir() -> str | None:
    f = os.path.join(BASE, ".last_dir.txt")
    if os.path.exists(f):
        try:
            return open(f, "r", encoding="utf-8").read().strip() or None
        except Exception:
            return None
    return None

def save_last_dir(p: str):
    try:
        d = p if os.path.isdir(p) else os.path.dirname(p)
        if d:
            with open(os.path.join(BASE, ".last_dir.txt"), "w", encoding="utf-8") as f:
                f.write(d)
    except Exception:
        pass

# ====== 3) LLM 誤字検出 ======
PROMPT_TMPL = """あなたはOCR後テキストの誤認識（誤字）検出・軽微修正を行う日本語の校正アシスタントです。
出力は必ず JSON だけにしてください：
{{
  "corrections": [{{"original":"誤認識らしき短いフレーズ","suggestion":"修正案","reason":"根拠"}}],
  "corrected_text": "行構造を概ね維持した全体の軽微修正文"
}}
対象テキスト:
<<<
{ocr_block}
>>>
注意: 0/O, 1/l/I, ー/一, 記号の混同に注意。確信の高い箇所だけ修正。最大8件。
"""

def chunk_text(s: str, max_chars: int = 800) -> List[str]:
    s = s.strip()
    if len(s) <= max_chars:
        return [s] if s else []
    chunks, buf, count = [], [], 0
    for line in s.splitlines():
        if count + len(line) + 1 > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, count = [], 0
        buf.append(line)
        count += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks

def llm_detect(pipe, ocr_block: str, max_new_tokens=120) -> Dict[str, Any]:
    import json, re
    prompt = PROMPT_TMPL.format(ocr_block=ocr_block)
    out = pipe.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8, top_p=0.95, top_k=50, repetition_penalty=1.1
    )
    m = re.search(r'\{.*\}\s*$', out, flags=re.S)
    if not m:
        return {"corrections": [], "corrected_text": ocr_block, "raw": out}
    try:
        obj = json.loads(m.group(0))
        if "corrections" in obj and "corrected_text" in obj:
            return obj
    except Exception:
        pass
    return {"corrections": [], "corrected_text": ocr_block, "raw": out}

# ====== 4) メイン ======
def main():
    ap = argparse.ArgumentParser(description="画像を選んでOCR→LLMで誤字検出")
    ap.add_argument("image", nargs="?", help="画像ファイル or フォルダ（省略時はダイアログ）")
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--device", default="NPU", choices=["NPU","CPU","GPU"])
    ap.add_argument("--model", default=r"C:\Hiroto\IR\models\TinyLlama-int4-ov")
    ap.add_argument("--out", default=r"C:\Hiroto\IR\result.json")
    ap.add_argument("--max_prompt", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    args = ap.parse_args()

    core = ov.Core()
    print("OV devices:", core.available_devices)

    # 画像リストを決定（1.引数ファイル 2.引数がフォルダ 3.ダイアログ）
    targets: list[str] = []
    if args.image:
        if os.path.isdir(args.image):
            patterns = ["*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"]
            for pat in patterns:
                targets += glob.glob(os.path.join(args.image, pat))
            targets.sort()
        elif os.path.isfile(args.image):
            targets = [args.image]
        else:
            print(f"指定パスが見つかりません: {args.image}", file=sys.stderr)
            sys.exit(1)
    else:
        last_dir = load_last_dir()
        picks = pick_images_with_dialog(last_dir)
        if not picks:
            print("画像が選択されませんでした。", file=sys.stderr)
            sys.exit(2)
        targets = list(picks)

    if not targets:
        print("処理対象の画像がありません。", file=sys.stderr)
        sys.exit(3)

    # EasyOCR
    backend, ocr = _load_ocr()
    print(f"[OCR] backend={backend}")

    pipe = load_llm(args.model, args.device, max_prompt_len=args.max_prompt, min_resp_len=16)

    all_reports = []
    t_all0 = time.time()

    for i, img in enumerate(targets, 1):
        save_last_dir(img)
        print(f"\n=== [{i}/{len(targets)}] {img} ===")
        t0 = time.time()
        try:
            o = run_ocr(backend, ocr, img)
            print(f"[OCR] lines={len(o['lines'])} chars={len(o['text'])} ({time.time()-t0:.2f}s)")
            blocks = chunk_text(o["text"], max_chars=800)
            all_corr, corr_blocks = [], []
            for bi, blk in enumerate(blocks, 1):
                print(f"[LLM] block {bi}/{len(blocks)} chars={len(blk)}")
                d = llm_detect(pipe, blk, max_new_tokens=args.max_new_tokens)
                all_corr.extend(d.get("corrections", []))
                corr_blocks.append(d.get("corrected_text", blk))
            corrected_text = "\n".join(corr_blocks)
            rep = {
                "image": os.path.abspath(img),
                "ocr_backend": backend,
                "ocr_lines": o["lines"],
                "ocr_text": o["text"],
                "llm_device": args.device,
                "llm_model": args.model,
                "corrections": all_corr,
                "corrected_text": corrected_text,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            all_reports.append(rep)
        except Exception as e:
            print(f"[ERROR] {img}: {type(e).__name__}: {e}")
            traceback.print_exc()

    # 結果保存（複数画像に対応）
    out_path = os.path.abspath(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"reports": all_reports, "total_sec": round(time.time()-t_all0, 2)},
            f, ensure_ascii=False, indent=2
        )
    print(f"\nSaved report: {out_path}")

if __name__ == "__main__":
    main()
