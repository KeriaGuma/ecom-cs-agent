"""ECom-Bench 工具服务（MCP）的包装：把识图工具接到阿里云百炼的视觉模型，然后原样启动 ECom-Bench 自带的服务。

ECom-Bench 开源时把识图模型的接口地址清空了。这里只替换“调用视觉模型”这一处，工具本身的提示词、
参数和其余 20 个工具都不变。

注意：这个进程的标准输出是 MCP 的通信通道，不能 print，日志只能写到标准错误。
MCP 子进程只继承少量环境变量，所以 Key 由本脚本自己从项目的 .env 读取，不通过命令行传入。
"""
import os
import re
import runpy
import sys
from pathlib import Path

from dotenv import dotenv_values

PROJECT = Path(__file__).resolve().parents[3]
ECOM_SERVER = PROJECT.parent / "ECom-Bench" / "agent" / "servers" / "offline"
VISION_MODEL = os.getenv("ECOM_VISION_MODEL", "qwen3.7-flash")  # 账号里没有 qwen3-vl-flash；qwen3.7-flash 原生支持图片，flash 档最便宜
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

env = dotenv_values(PROJECT / ".env")
VISION_KEY = env.get("DASHSCOPE_API_KEY") or env.get("EMBEDDING_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")

sys.path.insert(0, str(ECOM_SERVER))

import tools.apis.models.multimodal_model as mm  # noqa: E402
from openai import OpenAI  # noqa: E402
from openai.types.chat import ChatCompletionMessage  # noqa: E402

_client = None


def create_response(self, messages, model, tools=None, **kwargs):
    """替代原来的私有接口：文字 + 图片链接一起发给百炼的视觉模型（OpenAI 兼容接口）。"""
    global _client
    if _client is None:
        _client = OpenAI(api_key=VISION_KEY, base_url=DASHSCOPE_BASE)
    text = messages[-1]["content"]
    urls = list(dict.fromkeys(re.findall(self.url_pattern, text)))
    content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    try:
        r = _client.chat.completions.create(model=VISION_MODEL, temperature=0.0,
                                            messages=[{"role": "user", "content": content}])
        answer = r.choices[0].message.content or ""
    except Exception as e:  # 图片打不开或模型出错：如实告诉 Agent，不让工具崩溃
        print(f"[vision] {type(e).__name__}: {e}", file=sys.stderr)
        answer = "图片识别失败：无法读取图片内容。"
    return ChatCompletionMessage(content=answer, role="assistant")


mm.MultimodalModelClient.create_response = create_response

if __name__ == "__main__":
    runpy.run_path(str(ECOM_SERVER / "server.py"), run_name="__main__")
