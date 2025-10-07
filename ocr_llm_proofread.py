# ocr_llm_proofread.py  — GUIで画像選択 / EasyOCR固定 / NPU LLMで誤字検出
from string import Template
import os, sys, json, time, argparse, re, glob, textwrap, traceback, shutil
from typing import List, Dict, Any

# ====== 0) 環境パス（ASCII固定・一時/キャッシュ） ======
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

# ====== 1) 依存（LLM / EasyOCR / 画像処理） ======
import numpy as np
import cv2
import openvino as ov
import openvino_genai as ov_genai

# ----- NumPy混入をすべてPython標準型へ変換（JSON化のため） -----
def to_py(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    return obj

# ====== 2) EasyOCR  ======
def _load_ocr():
    import easyocr
    # 英数混在が多い場合は ['ja','en'] にしてもOK
    langs = ['ja']
    return ("easyocr", easyocr.Reader(langs))

def _to_box_py(box) -> list[list[int]]:
    arr = np.array(box, dtype=float).reshape(-1, 2)
    return [[int(round(x)), int(round(y))] for x, y in arr]

def run_ocr(backend: str, ocr, image_path: str) -> dict:
    """
    日本語パスでも確実に読み取れるように:
    - ファイルをバイナリで読み、imdecodeでndarray化
    - EasyOCRへndarrayを直接渡す
    - 見落とし対策: コントラスト強調・二値化の救済パス、画像が小さければ拡大パス
    - 近接ボックスの重複統合 + 読み順ソート
    """
    # --- Unicode安全読み込み ---
    img = None
    try:
        with open(image_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img = None

    def _easyocr(img_bgr):
        # 1) 原画そのまま
        res1 = ocr.readtext(
            img_bgr, detail=1, paragraph=False,
            low_text=0.2, text_threshold=0.5, link_threshold=0.3,
            mag_ratio=2.0, slope_ths=0.1, ycenter_ths=0.6, height_ths=0.6, width_ths=0.7
        )
        # 2) コントラスト強調＋二値化の救済パス
        g  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        g  = cv2.fastNlMeansDenoising(g, h=7)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g2 = clahe.apply(g)
        th = cv2.adaptiveThreshold(g2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
        th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        res2 = ocr.readtext(
            th, detail=1, paragraph=False,
            low_text=0.2, text_threshold=0.4, link_threshold=0.2,
            mag_ratio=2.0, slope_ths=0.2, ycenter_ths=0.7, height_ths=0.7, width_ths=0.8
        )
        return res1 + res2

    results = []
    if img is not None:
        results += _easyocr(img)
        # 小さい画像は拡大して再推論
        h, w = img.shape[:2]
        if (h < 600 or w < 600):
            img_up = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            results += _easyocr(img_up)
    else:
        # どうしても読めない場合はASCII一時コピー→パス入力で実行
        tmp_dir = os.environ.get("TMP", r"C:\Hiroto\IR\.tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, "ocr_work.png")
        shutil.copy2(image_path, tmp_path)
        results += ocr.readtext(tmp_path, detail=1, paragraph=False, mag_ratio=2.0)

    # --- 近接ボックスの重複統合（簡易） ---
    def _norm_xy(box):
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        return (int(round(sum(xs)/4)), int(round(sum(ys)/4)))

    merged = []
    seen   = []
    for (box, txt, conf) in results:
        if not str(txt).strip():
            continue
        box_py = _to_box_py(box)
        cx, cy = _norm_xy(box_py)
        dup = False
        for (px, py) in seen:
            if abs(cx - px) < 20 and abs(cy - py) < 20:
                dup = True
                break
        if not dup:
            seen.append((cx, cy))
            merged.append({"text": str(txt), "conf": float(conf), "box": box_py})

    # 読み順ソート（上→下、左→右）
    merged.sort(key=lambda r: (min(p[1] for p in r["box"]), min(p[0] for p in r["box"])))
    joined = "\n".join([m["text"] for m in merged])
    return {"lines": merged, "text": joined}

# ====== 3) LLM（OpenVINO GenAI, NPU/CPU/GPU） ======
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

PROMPT_TMPL = Template("""あなたはOCR後テキストの誤認識（誤字）検出・軽微修正を行う日本語の校正アシスタントです。
出力は必ず JSON だけにしてください：
{
  "corrections": [{"original":"誤認識らしき短いフレーズ","suggestion":"修正案","reason":"根拠"}],
  "corrected_text": "行構造を概ね維持した全体の軽微修正文"
}
対象テキスト:
<<<
$OCR_BLOCK
>>>
注意: 0/O, 1/l/I, ー/一, 記号の混同に注意。確信の高い箇所だけ修正。最大8件。
""")

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
    # Template で埋め込み（JSONの {} を誤解されない）
    prompt = PROMPT_TMPL.substitute(OCR_BLOCK=ocr_block)
    out = pipe.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8, top_p=0.95, top_k=50, repetition_penalty=1.1
    )
    # 末尾のJSONを抽出してパース（失敗時はrawを保持）
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
# ====== 4) GUI: ファイル選択 & 最終保存 ======
def pick_images_with_dialog(last_dir_hint: str | None = None) -> list[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return []
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

# ====== 5) メイン ======
def main():
    ap = argparse.ArgumentParser(description="画像を選んでOCR→LLMで誤字検出（NPU対応）")
    ap.add_argument("image", nargs="?", help="画像ファイル or フォルダ（省略時はダイアログ）")
    ap.add_argument("--device", default="NPU", choices=["NPU","CPU","GPU"])
    ap.add_argument("--model", default=r"C:\Hiroto\IR\models\TinyLlama-int4-ov")
    ap.add_argument("--out", default=r"C:\Hiroto\IR\result.json")
    ap.add_argument("--max_prompt", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    args = ap.parse_args()

    core = ov.Core()
    print("OV devices:", core.available_devices)

    # 画像リスト決定（引数ファイル/フォルダ/ダイアログ）
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

    # LLM
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

    # 結果保存（複数画像に対応）— NumPy型を全除去してから保存
    out_path = os.path.abspath(args.out)
    payload = {"reports": all_reports, "total_sec": round(time.time()-t_all0, 2)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_py(payload), f, ensure_ascii=False, indent=2)
    print(f"\nSaved report: {out_path}")

if __name__ == "__main__":
    main()
