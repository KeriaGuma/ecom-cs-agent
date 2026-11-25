"""回复检查的 LLM 复核：只改判“缺少关键词”，两个评委都同意才改判，引用必须真的出现在回复里。"""
import pytest

from cs_agent import reply_judge
from cs_agent.reply_judge import _verdict, missing_groups, recheck, regrade

TASK = {"id": "t1", "turns": ["键盘取消好了吗"],
        "expect": {"reply": [["已取消", "已退款"], ["459"]]}}
REPLY = "键盘订单的退款已处理完成，459 元已原路退回。"
FAILS = ["回复：缺少 已取消/已退款", "关键路径：没有调用 get_order"]


@pytest.fixture
def judges(monkeypatch):
    """按评委名设定结论，记录调用次数。"""
    said, calls = {}, []

    def fake_ask(judge, turns, reply, group, retries=3):
        calls.append((judge, tuple(group)))
        return _verdict({"point": "已退款", "evidence": "退款已处理完成", "conveyed": said[judge]}, reply)

    monkeypatch.setattr(reply_judge, "ask", fake_ask)
    return said, calls


def test_only_missing_groups_are_sent_to_judges(judges):
    said, calls = judges
    said.update(qwen=True, kimi=True)
    assert missing_groups(TASK["expect"], REPLY) == [["已取消", "已退款"]]  # “459”命中了，不复核
    fails, records = recheck(TASK, ["…", REPLY], FAILS)
    assert {g for _, g in calls} == {("已取消", "已退款")}
    assert fails == ["关键路径：没有调用 get_order"]  # 其他失败原样保留
    assert records[0]["conveyed"]


@pytest.mark.parametrize("qwen,kimi", [(True, False), (False, True), (False, False)])
def test_both_judges_must_agree_to_flip(judges, qwen, kimi):
    said, _ = judges
    said.update(qwen=qwen, kimi=kimi)
    fails, _ = recheck(TASK, [REPLY], FAILS)
    assert fails == FAILS


def test_evidence_must_be_quoted_from_reply():
    assert _verdict({"conveyed": True, "evidence": "退款已处理完成"}, REPLY)["conveyed"]
    assert not _verdict({"conveyed": True, "evidence": "订单已取消"}, REPLY)["conveyed"]  # 回复里没有这句
    assert not _verdict({"conveyed": True, "evidence": ""}, REPLY)["conveyed"]
    # 评委摘抄时常把中文引号换成英文引号、丢掉 Markdown 加粗：不算编造
    assert _verdict({"conveyed": True, "evidence": '状态显示为"退款成功"'}, "状态显示为**“退款成功”**")["conveyed"]
    with pytest.raises(ValueError):
        _verdict({"evidence": "x"}, REPLY)


def test_regrade_keeps_keyword_result_and_clears_severity(judges):
    said, _ = judges
    said.update(qwen=True, kimi=True)
    only_reply = {"id": "t1", "replies": [REPLY], "failures": ["回复：缺少 已取消/已退款"], "success": False,
                  "severity": "沟通"}
    mixed = {**only_reply, "failures": FAILS, "severity": "流程"}
    ok = {"id": "t1", "replies": [REPLY], "failures": [], "success": True, "severity": None}
    out = regrade([only_reply, mixed, ok], {"t1": TASK}, workers=1)
    assert out[0]["success"] and out[0]["severity"] is None
    assert out[0]["keyword_failures"] == ["回复：缺少 已取消/已退款"]
    assert not out[1]["success"] and out[1]["severity"] == "流程"
    assert out[2] is ok  # 没有回复检查失败的不动


def test_forbidden_phrases_are_never_rechecked(judges):
    said, calls = judges
    said.update(qwen=True, kimi=True)
    task = {"id": "t2", "turns": ["x"], "expect": {"reply_forbidden": ["O2001"]}}
    fails, records = recheck(task, ["您的订单 O2001"], ["回复：不应出现 O2001"])
    assert fails == ["回复：不应出现 O2001"] and not records and not calls
