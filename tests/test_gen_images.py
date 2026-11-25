"""生图脚本：解析提示词文档；免费额度用完时换下一个模型（用假接口，不真的调用）。"""
import pytest

from cs_agent.gen_images import AccountError, ModelPool, generate_any, parse_doc

DOC = """# 标题
### d01.jpg：裂纹
```
提示词一
```
必须能看到：裂纹
## t02.jpg
- 说明
```
提示词二
```
"""


class Resp:
    def __init__(self, status, data):
        self.status_code, self._data = status, data

    def json(self):
        return self._data


def ok(url="http://img/1.png"):
    return Resp(200, {"output": {"choices": [{"message": {"content": [{"image": url}]}}]}})


def err(status, code):
    return Resp(status, {"code": code, "message": "x"})


QUOTA = err(403, "AllocationQuota.FreeTierOnly")


class FakeApi:
    """按模型给出预设的回复序列，记录调用了哪些模型。"""

    def __init__(self, script):
        self.script = {m: list(rs) for m, rs in script.items()}
        self.calls = []

    def __call__(self, body, key, timeout):
        self.calls.append(body["model"])
        return self.script[body["model"]].pop(0)


def run(script, models, n=1):
    api, pool = FakeApi(script), ModelPool(models)
    out = [generate_any("p", pool, "k", post=api, sleep=lambda s: None) for _ in range(n)]
    return out, api.calls, pool


def test_parse_doc_handles_both_heading_styles():
    assert parse_doc(DOC) == {"d01": "提示词一", "t02": "提示词二"}


def test_switches_model_when_free_quota_is_used_up():
    out, calls, pool = run({"a": [ok("u1"), QUOTA], "b": [ok("u2"), ok("u3")]}, ["a", "b"], n=3)
    assert out == [("u1", "a"), ("u2", "b"), ("u3", "b")]
    assert calls == ["a", "a", "b", "b"]  # a 用完后不再回头试 a
    assert list(pool.gone) == ["a"]


def test_single_access_denied_is_retried_not_switched():
    out, calls, _ = run({"a": [err(403, "Workspace.AccessDenied"), ok()]}, ["a", "b"])
    assert out[0][1] == "a" and calls == ["a", "a"]


def test_repeated_access_denied_switches_model():
    out, _, pool = run({"a": [err(403, "Workspace.AccessDenied")] * 2, "b": [ok()]}, ["a", "b"])
    assert out[0][1] == "b" and "a" in pool.gone


def test_rate_limit_waits_and_retries_same_model():
    out, calls, _ = run({"a": [err(429, "Throttling.RateQuota"), ok()]}, ["a", "b"])
    assert out[0][1] == "a" and calls == ["a", "a"]


def test_stops_when_every_model_is_used_up():
    with pytest.raises(AccountError, match="都用完了"):
        run({"a": [QUOTA], "b": [QUOTA]}, ["a", "b"])


def test_account_problem_stops_without_trying_other_models():
    api = FakeApi({"a": [err(400, "Arrearage")], "b": [ok()]})
    with pytest.raises(AccountError):
        generate_any("p", ModelPool(["a", "b"]), "k", post=api, sleep=lambda s: None)
    assert api.calls == ["a"]


def test_other_errors_fail_the_image_but_keep_the_model():
    api, pool = FakeApi({"a": [err(400, "DataInspectionFailed")]}), ModelPool(["a", "b"])
    with pytest.raises(RuntimeError):
        generate_any("p", pool, "k", post=api, sleep=lambda s: None)
    assert pool.gone == {}  # 内容审核之类的问题换模型也没用，不停用模型
