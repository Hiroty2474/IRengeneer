import os, sys, traceback, glob
from huggingface_hub import snapshot_download
import openvino as ov
import openvino_genai as ov_genai

BASE = r"C:\Hiroto\IR"
HF_HOME  = os.path.join(BASE, ".hf")
HF_HUB   = os.path.join(HF_HOME, "hub")
OV_CACHE = os.path.join(BASE, ".ovcache")
TMP_DIR  = os.path.join(BASE, ".tmp")
MODEL_DIR = os.path.join(BASE, "models", "TinyLlama-int4-ov")

for p in (HF_HOME, HF_HUB, OV_CACHE, TMP_DIR, MODEL_DIR):
    os.makedirs(p, exist_ok=True)

# ===== すべて ASCII パスに固定 =====
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"]  = TMP_DIR
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HUB
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "tf")
os.environ["OPENVINO_CACHE_DIR"] = OV_CACHE
# 警告を消すだけ（動作には不要）
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 失敗原因を出すため詳細ログ（うるさければ削除OK）
os.environ.setdefault("OV_LOG_LEVEL", "DEBUG")

core = ov.Core()
print("OV available devices:", core.available_devices)

REPO_ID = "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"

print(f"\n[DL] Downloading to local_dir (copy mode): {MODEL_DIR}")
# ★ ここがキモ：symlink を使わずコピーに切替
model_dir = snapshot_download(
    REPO_ID,
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False,   # <= 重要！
    resume_download=True,
)

print("[DL] model_dir:", model_dir)

# 中身チェック（tokenizer等）
need_globs = [
    "**/*.xml", "**/*.bin",
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "vocab.json", "merges.txt", "config.json", "generation_config.json",
    "special_tokens_map.json"
]
for pat in need_globs:
    found = glob.glob(os.path.join(model_dir, pat), recursive=True)
    print(f"[CHK] {pat}: {('FOUND ' + str(len(found))) if found else 'MISSING'}")
    for f in found[:5]:
        print("   ", os.path.relpath(f, model_dir))
    if len(found) > 5:
        print("    ... (+%d more)" % (len(found)-5))

def try_build(device, **kwargs):
    print(f"\n[INIT] device={device} kwargs={kwargs}")
    try:
        pipe = ov_genai.LLMPipeline(
            model_dir,
            device,
            CACHE_DIR=OV_CACHE,
            GENERATE_HINT="BEST_PERF",
            MAX_PROMPT_LEN=512,
            MIN_RESPONSE_LEN=32,
            **kwargs
        )
        print("[INIT] OK on", device)
        return pipe
    except Exception as e:
        print(f"[INIT] FAIL on {device}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

# NPU → NPU(L0無効) → CPU の順に試す
pipe = try_build("NPU")
if pipe is None:
    os.environ["DISABLE_OPENVINO_GENAI_NPU_L0"] = "1"
    pipe = try_build("NPU", note="L0_DISABLED")
    os.environ.pop("DISABLE_OPENVINO_GENAI_NPU_L0", None)
if pipe is None:
    pipe = try_build("CPU")

if pipe is None:
    print("\n=== Init failed on all devices. 上の [CHK] / [INIT] の出力を貼ってください。===")
    sys.exit(1)

# 動作テスト
prompt = "自己紹介を、箇条書きで3点、短く。"
print("\n[RUN] Prompt:", prompt)
out = pipe.generate(prompt, max_new_tokens=100)
print("\n=== Output ===")
print(out)
