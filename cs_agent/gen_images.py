"""按生图提示词文档批量生成评测图片（阿里云百炼的文生图接口）。

    python -m cs_agent.gen_images --doc eval/images/IMAGE_PROMPTS_DEV.md --out eval/images/dev
    python -m cs_agent.gen_images --doc ... --out ... --only d04 d14 --overwrite   # 不合格的重新生成

- 文档格式：标题行里写文件名（“### d01.jpg：……”或“## t01.jpg”），标题后的第一个代码块就是提示词；
- 关闭提示词改写（prompt_extend=false），提示词原样交给模型，图片要点和文档一致；
- 存成 JPG，长边压到 1536，接近手机照片的大小；
- **按顺序轮换模型**：当前模型的免费额度用完（接口返回 AllocationQuota.FreeTierOnly），就换列表里的下一个，
  所有模型都用完才停。前提是控制台里给这些模型开了“用完即停”，否则额度用完后接口照常出图、按量计费，
  脚本察觉不到；
- 生图模型不肯把主体拍糊（实测要求“整张拍糊”时只会虚化背景），需要“拍糊了”的图用 --blur 名字=半径
  在生成后加高斯模糊，半径记进生成记录；
- 已经存在的图片默认跳过，生成记录（用的哪个模型、提示词摘要、时间）写进输出目录的 _manifest.json；
- Key 只从 .env 读（DASHSCOPE_API_KEY），不打印。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .config import ROOT

URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
HEAD = re.compile(r"^#{2,3}\s*([a-z]\d+)\.(?:jpg|png)", re.M)
MAX_SIDE = 1536
# 优先级：先用实测效果好的（中文字、裂纹都画得准），z-image-turbo 裂纹偏淡，放最后
MODELS = ("qwen-image-3.0", "qwen-image-3.0-pro", "qwen-image-2.1-pro", "qwen-image-2.0-pro",
          "wan2.7-image", "wan2.7-image-pro", "qwen-image-2.0", "z-image-turbo")
# 这个模型不能再用了：免费额度用完（开了“用完即停”）、没开通、不支持这个接口
MODEL_GONE = ("AllocationQuota", "FreeTierOnly", "Model.AccessDenied", "ModelNotFound", "Model.NotSupport")
ACCOUNT_STOP = ("Arrearage", "InvalidApiKey")  # 账号问题，换模型也没用


class ModelGone(Exception):
    pass


class AccountError(Exception):
    pass


def parse_doc(text: str) -> dict[str, str]:
    """{图片名: 提示词}。每个标题之后、下一个标题之前的第一个代码块。"""
    heads = list(HEAD.finditer(text))
    out = {}
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        block = re.search(r"```[^\n]*\n(.*?)\n```", text[h.end():end], re.S)
        if block and h.group(1) not in out:
            out[h.group(1)] = block.group(1).strip()
    return out


def _post(body: dict, key: str, timeout: float) -> httpx.Response:
    return httpx.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=timeout)


def generate(prompt: str, model: str, key: str, timeout: float = 180, post=_post, sleep=time.sleep) -> str:
    """返回图片下载地址。额度用完等“这个模型不能用了”抛 ModelGone，账号问题抛 AccountError。"""
    body = {"model": model, "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
            "parameters": {"prompt_extend": False, "watermark": False}}
    access_denied = 0
    for attempt in range(8):
        r = post(body, key, timeout)
        if r.status_code == 429:  # 限流：等一会儿再试
            sleep(10 * (attempt + 1))
            continue
        data = r.json()
        code = str(data.get("code") or "")
        try:
            return next(c["image"] for c in data["output"]["choices"][0]["message"]["content"] if "image" in c)
        except Exception:
            pass
        msg = f"{model}：HTTP {r.status_code} {code} {data.get('message')}"
        if any(k in code for k in ACCOUNT_STOP):
            raise AccountError(msg)
        if any(k in code for k in MODEL_GONE):
            raise ModelGone(msg)
        if code == "Workspace.AccessDenied":  # 实测偶尔会误报，重试一次；连续两次才认为这个模型用不了
            access_denied += 1
            if access_denied >= 2:
                raise ModelGone(msg)
            sleep(5)
            continue
        raise RuntimeError(msg)
    raise RuntimeError(f"{model}：一直被限流")


class ModelPool:
    """按顺序轮换的模型列表，多个线程共用：一个模型被判定用完后，所有线程都不再用它。"""

    def __init__(self, models):
        self.models = list(models)
        self.gone: dict[str, str] = {}
        self.lock = threading.Lock()

    def current(self):
        with self.lock:
            return next((m for m in self.models if m not in self.gone), None)

    def drop(self, model: str, reason: str) -> None:
        with self.lock:
            if model not in self.gone:
                self.gone[model] = reason
                print(f"  ⚠ {reason}\n    → 不再使用 {model}，换下一个模型")


def generate_any(prompt: str, pool: ModelPool, key: str, **kw) -> tuple[str, str]:
    """用池子里当前可用的模型生成，用完了就换下一个。返回 (图片地址, 实际用的模型)。"""
    while True:
        model = pool.current()
        if model is None:
            raise AccountError("所有模型的免费额度都用完了（或都不可用）：" + "；".join(pool.gone.values()))
        try:
            return generate(prompt, model, key, **kw), model
        except ModelGone as e:
            pool.drop(model, str(e))


def to_jpeg(raw: bytes, blur: float = 0) -> bytes:
    from PIL import Image, ImageFilter

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description="按提示词文档生成评测图片")
    ap.add_argument("--doc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS), help="按顺序使用，额度用完换下一个")
    ap.add_argument("--only", nargs="*", help="只生成这几张，比如 d04 d14")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已有图片（重新生成不合格的图）")
    ap.add_argument("--blur", nargs="*", default=[], help="生成后加模糊，比如 d04=18（半径按长边 1536 像素算）")
    ap.add_argument("--workers", type=int, default=2, help="并发数；生图接口限流较严，默认 2")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = os.getenv("DASHSCOPE_API_KEY", "")
    if not key:
        raise SystemExit("请在 .env 里设置 DASHSCOPE_API_KEY")

    prompts = parse_doc(Path(args.doc).read_text())
    if args.only:
        missing = set(args.only) - set(prompts)
        if missing:
            raise SystemExit(f"文档里没有这些图片：{sorted(missing)}")
        prompts = {k: v for k, v in prompts.items() if k in args.only}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = {k: v for k, v in prompts.items() if args.overwrite or not list(out.glob(f"{k}.*"))}
    pool = ModelPool(args.models)
    blur = {k: float(v) for k, v in (b.split("=") for b in args.blur)}
    print(f"文档里 {len(prompts)} 张，本次生成 {len(todo)} 张；模型顺序：{' → '.join(pool.models)}")

    manifest_path = out / "_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    stop = threading.Event()

    def one(item):
        name, prompt = item
        if stop.is_set():
            return name, None, "已停止"
        t0 = time.time()
        try:
            url, model = generate_any(prompt, pool, key)
            data = to_jpeg(httpx.get(url, timeout=60).content, blur.get(name, 0))
        except AccountError as e:
            stop.set()
            return name, None, str(e)
        except Exception as e:
            return name, None, str(e)
        for old in out.glob(f"{name}.*"):
            old.unlink()
        (out / f"{name}.jpg").write_bytes(data)
        return name, {"model": model, "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:8],
                      **({"blur_radius": blur[name]} if name in blur else {}),
                      "seconds": round(time.time() - t0, 1), "at": time.strftime("%Y-%m-%d %H:%M:%S")}, None

    failed = []
    with ThreadPoolExecutor(args.workers) as ex:
        for name, rec, err in ex.map(one, sorted(todo.items())):
            if err:
                failed.append(name)
                print(f"  {name} 失败：{err}")
            else:
                manifest[name] = rec
                print(f"  {name} 完成（{rec['model']}，{rec['seconds']}s）")
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), ensure_ascii=False, indent=1))
    if pool.gone:
        print("本次停用的模型：" + "、".join(pool.gone))
    if failed:
        raise SystemExit(f"{len(failed)} 张没生成：{' '.join(failed)}")


if __name__ == "__main__":
    main()
