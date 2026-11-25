"""沟通质量评委的离线部分：对话渲染、输出解析、一致性统计、缺陷植入（不调用模型）。"""
import pytest

from cs_agent.eval_judge import DIMS, cohen_kappa, corrupt_fact, paired, parse_json, render_transcript, validate


def _scores(**kw):
    base = {d: {"score": top, "evidence": "无问题"} for d, top in DIMS.items()}
    for d, s in kw.items():
        base[d] = {"score": s, "evidence": "x"}
    return base


def test_parse_json_strips_code_fence_and_chatter():
    assert parse_json('好的：\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("没有 JSON")


def test_validate_rejects_missing_or_out_of_range():
    assert validate(_scores())["concise"]["score"] == 2
    bad = _scores()
    del bad["tone"]
    with pytest.raises(ValueError):
        validate(bad)
    with pytest.raises(ValueError):
        validate(_scores(answered=2))  # 0/1 维度给了 2 分


def test_render_shows_approval_event_only_when_refund_went_to_review():
    tools = [[{"tool": "submit_refund", "args": {}, "output": "退款申请已提交，金额 1899.00 元超过自动审批额度，需人工审核"}], []]
    text = render_transcript(["退了", "谢谢"], ["已提交", "不客气"], tools, approval="reject")
    assert "审核员已驳回" in text and text.count("系统事件") == 1
    assert "（本轮没有调用工具）" in text
    small = [[{"tool": "submit_refund", "args": {}, "output": "退款成功：199.00 元将原路退回"}]]
    assert "系统事件" not in render_transcript(["退了"], ["好的"], small)


def test_cohen_kappa():
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) is None  # 只有一种取值时无定义
    assert cohen_kappa([2, 1, 0], [2, 1, 1], weighted=True) == pytest.approx(0.5714, abs=1e-3)


def test_corrupt_fact_prefers_amount_found_in_tool_output():
    tools = [{"output": "11-10 到达；refund_amount 199.0"}]
    assert corrupt_fact("11-10 到达，可退 199 元", tools) == "11-10 到达，可退 299 元"
    assert corrupt_fact("运费由您承担", [{"output": ""}]) == "运费由商家承担"
    assert corrupt_fact("好的", [{"output": ""}]) is None


def test_paired_difference_per_task():
    rows = []
    for tid, m, s in (("t1", 1, 0), ("t2", 1, 1)):
        rows.append({"id": tid, "system": "multi", "scores": {"k": {0: _scores(faithful=m)}}})
        rows.append({"id": tid, "system": "single", "scores": {"k": {0: _scores(faithful=s)}}})
    out = paired(rows, "k")
    assert out["faithful"]["diff"] == 0.5 and out["faithful"]["multi_better"] == 1
