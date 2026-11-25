"""看图准确率：只评视觉模型这一步，不跑整个客服流程。

    python -m cs_agent.eval_vision --tasks eval/e2e_image_dev.jsonl --models qwen3-vl-plus qwen3.7-flash

标准答案来自任务里的 image_label（生成图片时就定好的）。每张图只评一次（同一张图被多个任务引用时去重）。

指标：
- usable：照片清不清楚、能不能作为依据，判对的比例；
- issue：外观问题“有 / 没有”判对的比例（只看清楚的照片）；
- false_issue：外观完好的照片被说成有问题的比例。这是最要紧的错误：会被当成质量问题，退货运费变成商家出；
- item：物品类别认对的比例（按关键词归类：耳机、扫地机器人、保温杯、鞋、截图、包装盒）。
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from .config import ROOT, settings
from .eval_e2e import find_image, load_tasks
from .vision import get_vlm, inspect_photo

CATEGORIES = {"耳机": ("耳机", "充电仓", "耳塞"), "扫地机器人": ("扫地", "机器人"), "保温杯": ("保温杯", "杯"),
              "鞋": ("鞋",), "截图": ("截图", "屏幕", "界面"), "包装盒": ("包装", "纸盒", "盒子")}
MODEL_EXTRA = {"qwen3.7-flash": {"enable_thinking": False}}  # 关闭思考，和其他模型同口径


def category(text: str) -> str:
    for name, keys in CATEGORIES.items():  # 顺序有讲究：“耳机充电仓”先归到耳机
        if any(k in text for k in keys):
            return name
    return "其他"


def expected(label: dict) -> dict:
    issue = str(label.get("visible_issue", ""))
    has_issue = None if not label.get("usable", True) else not issue.startswith("无")
    return {"usable": bool(label.get("usable", True)), "has_issue": has_issue, "item": category(str(label.get("item", "")))}


def images_of(tasks: list[dict], image_dir: Path) -> list[dict]:
    seen, out = set(), []
    for t in tasks:
        first = t["turns"][0]
        if not isinstance(first, dict) or not first.get("images") or "image_label" not in t:
            continue
        name = first["images"][0]
        if name in seen:
            continue
        seen.add(name)
        out.append({"image": name, "path": str(find_image(image_dir, name)), "label": t["image_label"],
                    "task": t["id"]})
    return out


def score(pred: dict, exp: dict) -> dict:
    has_issue = pred["visible_issue"] not in ("无", "看不清")
    out = {"usable_ok": pred["usable"] == exp["usable"], "item_ok": category(pred["item"]) == exp["item"]}
    if exp["has_issue"] is not None and pred["usable"]:
        out["issue_ok"] = has_issue == exp["has_issue"]
        if not exp["has_issue"]:
            out["false_issue"] = has_issue
    return out


def run_model(model: str, items: list[dict], repeat: int, workers: int) -> dict:
    vlm = get_vlm(replace(settings, vision=True, vision_model=model, vision_extra_body=MODEL_EXTRA.get(model, {})))
    if vlm is None:
        raise SystemExit("没有视觉模型的 Key：请在 .env 里设置 DASHSCOPE_API_KEY 或 VISION_API_KEY")

    def one(job):
        item, r = job
        t0 = time.perf_counter()
        try:
            pred = inspect_photo(vlm, item["path"])
        except Exception as e:
            return {**item, "run": r, "error": str(e)}
        return {**item, "run": r, "pred": pred, "latency_ms": (time.perf_counter() - t0) * 1000,
                **score(pred, expected(item["label"]))}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, [(it, r) for it in items for r in range(repeat)]))
    ok = [x for x in rows if "pred" in x]

    def rate(key):
        xs = [x[key] for x in ok if key in x]
        return {"rate": round(sum(xs) / len(xs), 3) if xs else None, "n": len(xs)}

    lat = sorted(x["latency_ms"] for x in ok)
    return {"model": model, "errors": len(rows) - len(ok), "usable": rate("usable_ok"), "issue": rate("issue_ok"),
            "false_issue": rate("false_issue"), "item": rate("item_ok"),
            "latency_p50_ms": round(lat[len(lat) // 2]) if lat else None, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="看图准确率")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--image-dir")
    ap.add_argument("--models", nargs="+", default=[settings.vision_model])
    ap.add_argument("--repeat", type=int, default=2, help="每张图重复识别几次（看稳定性）")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out")
    args = ap.parse_args()

    path = Path(args.tasks)
    image_dir = Path(args.image_dir) if args.image_dir else ROOT / "eval" / "images" / ("test" if "image_test" in path.name else "dev")
    items = images_of(load_tasks(path), image_dir)
    report = [run_model(m, items, args.repeat, args.workers) for m in args.models]
    for r in report:
        print(f"{r['model']:22s} 清楚与否 {r['usable']['rate']}  外观问题 {r['issue']['rate']}  "
              f"完好被说成有问题 {r['false_issue']['rate']}(n={r['false_issue']['n']})  物品 {r['item']['rate']}  "
              f"p50 {r['latency_p50_ms']}ms  出错 {r['errors']}")
        for x in r["rows"]:
            bad = [k for k in ("usable_ok", "issue_ok", "item_ok") if x.get(k) is False] + (["false_issue"] if x.get("false_issue") else [])
            if bad or "error" in x:
                p = x.get("pred") or {}
                print(f"    {x['image']}#{x['run']} {bad or x.get('error')}  标注={x['label']}  "
                      f"识别={p.get('item')}/{p.get('usable')}/{p.get('visible_issue')}/{p.get('detail', '')[:30]}")
    if args.out:
        Path(args.out).write_text(json.dumps({"tasks": str(path), "images": len(items), "report": report},
                                             ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
