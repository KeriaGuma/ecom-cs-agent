"""ECom-Bench 启动脚本（在 ECom-Bench 的虚拟环境里运行，不修改 ECom-Bench 源码）。

ECom-Bench 开源时清空了所有模型接口，这里在运行时注入：
- 客服和模拟用户：DeepSeek（OpenAI 兼容接口，关闭思考模式）。客服 temperature=0（与本项目其他实验一致），
  模拟用户保持 ECom-Bench 的默认 0.3；
- 识图工具：见 server/server.py，接阿里云百炼的 qwen3.7-flash；
- 客服实现：--impl baseline 复现 ECom-Bench 自带的单 Agent，single 为工程化单 Agent，team 为 Supervisor 多 Agent 对照。
Key 只从本项目 .env 读出，不出现在命令行参数和结果文件里。

用法：
  ../ECom-Bench/.venv/bin/python bench/ecom/run.py --impl baseline --split dev --num-trials 1 --log-dir <目录>
  （其余参数原样交给 ECom-Bench 的 run.py，如 --task-ids、--max-concurrency）
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ECOM = PROJECT.parent / "ECom-Bench"

env = dotenv_values(PROJECT / ".env")
KEY = env.get("LLM_API_KEY") or os.getenv("LLM_API_KEY")
if not KEY:
    raise SystemExit("未在 .env 中找到 LLM_API_KEY")
BASE = env.get("LLM_BASE_URL") or "https://api.deepseek.com"
MODEL = env.get("WORKER_MODEL") or "deepseek-flash"
NO_THINKING = {"thinking": {"type": "disabled"}}
TEMPERATURE = {"ds-agent": 0.0, "ds-user": 0.3}

ours = argparse.ArgumentParser(add_help=False)
ours.add_argument("--impl", choices=["baseline", "single", "team"], default="baseline")
ours.add_argument("--split", choices=["dev", "test"])
args, rest = ours.parse_known_args()

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ECOM))
os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"  # MCP 子进程用 python 启动

import time  # noqa: E402

import utils  # noqa: E402  (ECom-Bench 的 utils)
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402


class AgentUsage(BaseCallbackHandler):
    """统计客服一侧（不含模拟用户）的模型调用次数和 Token；每种实现单独一个进程，所以按进程累计即可。"""

    def __init__(self):
        self.calls = self.input_tokens = self.output_tokens = 0
        self.turn_seconds: list[float] = []

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                um = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.calls += 1
                self.input_tokens += um.get("input_tokens", 0)
                self.output_tokens += um.get("output_tokens", 0)


USAGE = AgentUsage()


def make_llm(temperature: float, agent: bool) -> ChatOpenAI:
    return ChatOpenAI(base_url=BASE, api_key=KEY, model=MODEL, temperature=temperature, extra_body=NO_THINKING,
                      callbacks=[USAGE] if agent else None)


def _initiate_llm(self):
    if self.model_name not in TEMPERATURE:
        raise ValueError(f"只支持 {list(TEMPERATURE)}，收到 {self.model_name!r}")
    self.llm = make_llm(TEMPERATURE[self.model_name], agent=self.model_name == "ds-agent")


utils.LLM._initiate_llm = _initiate_llm

import envs.story.env as story_env  # noqa: E402

story_env.OFFLINE_SERVER_DIR = str(HERE / "server")  # 用包装过的工具服务（识图接百炼）

import agent as ecom_agent  # noqa: E402

if args.impl != "baseline":
    from bench.ecom.ecom_single import AGENTS as SINGLE_AGENTS  # noqa: E402
    from bench.ecom.ecom_team import AGENTS  # noqa: E402
    ecom_agent.AgentLangChain = {**AGENTS, **SINGLE_AGENTS}[args.impl]  # env.py 在运行时 from agent import AgentLangChain

_orig_call = ecom_agent.AgentLangChain.call


async def _timed_call(self, message):  # 记录客服每轮回复的耗时
    t = time.perf_counter()
    try:
        return await _orig_call(self, message)
    finally:
        USAGE.turn_seconds.append(round(time.perf_counter() - t, 3))


ecom_agent.AgentLangChain.call = _timed_call

if args.split:
    ids = json.load(open(HERE / "split.json"))[args.split]
    rest += ["--task-ids", *map(str, ids)]

sys.argv = [sys.argv[0], "--user-model", "ds-user", "--agent-model", "ds-agent", *rest]

from main import run  # noqa: E402
from run import parse_args  # noqa: E402  (ECom-Bench 的 run.py)

if __name__ == "__main__":
    config = parse_args()
    asyncio.run(run(config))
    os.makedirs(config.log_dir, exist_ok=True)
    with open(Path(config.log_dir) / f"usage_{args.impl}.json", "w") as f:
        runtime = "upstream" if args.impl == "baseline" else "python-async"
        json.dump({"impl": args.impl, "runtime": runtime, "model": MODEL, "calls": USAGE.calls, "input_tokens": USAGE.input_tokens,
                   "output_tokens": USAGE.output_tokens, "turn_seconds": USAGE.turn_seconds}, f)
