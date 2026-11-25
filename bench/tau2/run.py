"""τ²-bench 启动脚本（在 tau2-bench 的虚拟环境里运行）。

DeepSeek 的 Key 只从本项目 .env 读出来放进环境变量，不出现在命令行参数里，
也就不会被 τ²-bench 记录到结果文件中（它会把模型参数原样写进结果）。

用法：
  ../tau2-bench/.venv/bin/python bench/tau2/run.py run --domain retail ...
"""
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

PROJECT = Path(__file__).resolve().parents[2]
TAU2_HOME = PROJECT.parent / "tau2-bench"

env = dotenv_values(PROJECT / ".env")
key = env.get("LLM_API_KEY") or os.getenv("LLM_API_KEY")
if not key:
    raise SystemExit("未在 .env 中找到 LLM_API_KEY")
os.environ["DEEPSEEK_API_KEY"] = key
os.environ["DEEPSEEK_API_BASE"] = env.get("LLM_BASE_URL") or "https://api.deepseek.com"
os.environ.setdefault("TAU2_DATA_DIR", str(TAU2_HOME / "data"))

# DeepSeek 默认开思考模式；带工具的多轮调用要求回传 reasoning_content，通用客户端不回传，所以关掉
NO_THINKING = {"thinking": {"type": "disabled"}}
MODEL = "deepseek/" + (env.get("WORKER_MODEL") or "deepseek-flash")

sys.path.insert(0, str(PROJECT))  # 让 bench.tau2.* 里的自定义 Agent 可以被导入注册

from tau2.cli import main  # noqa: E402
import tau2.evaluator.evaluator_nl_assertions as _nl  # noqa: E402

# 回复要点判定（NL assertions）的模型在 τ²-bench 源码里写死为 OpenAI 的 gpt-5.4-mini，没有命令行参数可改。
# 这里换成同一个 DeepSeek 模型，不改 τ²-bench 源码。报告结果时需注明“判定模型与官方不同”。
_nl.DEFAULT_LLM_NL_ASSERTIONS = MODEL
_nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 0.0, "extra_body": NO_THINKING}

# 注册我们的 Supervisor 多 Agent（--agent supervisor_agent）；单 Agent 基线直接用 τ²-bench 自带的 llm_agent
from tau2.registry import registry  # noqa: E402

from bench.tau2.supervisor_agent import create_supervisor_agent  # noqa: E402

registry.register_agent_factory(create_supervisor_agent, "supervisor_agent")

# “调度员规划、Worker 汇报”的多 Agent（--agent plan_agent），结构和本项目 cs_agent 一致
from bench.tau2.plan_agent import create_plan_agent  # noqa: E402

registry.register_agent_factory(create_plan_agent, "plan_agent")

# 零售 + 航空通用的调度员汇报版：team_relay（转述确认）、team_struct（结构化方案，确认后由代码原样执行）
from bench.tau2.team_agent import create_team_relay, create_team_struct  # noqa: E402

registry.register_agent_factory(create_team_relay, "team_relay")
registry.register_agent_factory(create_team_struct, "team_struct")

# “是否告知了用户某条信息”（communicate_info，航空领域会用到）的判定模型默认是 OpenAI 的 gpt-5.5，
# 同样换成 DeepSeek；通过 τ²-bench 自带的 --communicate-judge 参数传入，不改源码。
COMMUNICATE_JUDGE = json.dumps({"model": MODEL, "model_args": {"temperature": 0.0, "extra_body": NO_THINKING}})

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run" and "--communicate-judge" not in sys.argv:
        sys.argv += ["--communicate-judge", COMMUNICATE_JUDGE]
    sys.exit(main())
