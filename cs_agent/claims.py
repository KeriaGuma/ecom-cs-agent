"""“谎报完成”代码守卫：回复声称已经办好或已生成确认单，但本轮处理未调用对应写工具。

只靠系统提示词约束模型“先调工具再说办好”在多意图场景下存在漏调风险：
例如用户一句话同时要求“帮我退掉耳机，顺便问下几天到账”，模型查完到账政策后可能直接顺口回复“已为您提交退款”。
因此在 ReAct 循环出口通过确定性代码校验写工具调用轨迹，发现无依据承诺时静默打回重跑一次。
"""
from __future__ import annotations

import re

# 会产生业务状态变更或确认单草稿的工具（转人工由专用控制流处理）
WRITE_TOOLS = frozenset({
    "prepare_refund", "submit_refund", "change_address", "reissue_points", "exchange_points",
    "issue_invoice", "reissue_invoice", "return_coupon", "apply_repair", "apply_price_protect",
})

CLAIM = re.compile(
    r"已(?:经)?(?:为您|帮您|给您)?(?:成功)?(?:提交|办理|生成|创建|发起|退款|退回|修改|改好|取消|处理好)"
    r"(?:了|好)?(?:退款|退货|确认单|申请|地址|订单)?"
    r"|(?:退款|退货|确认单|修改|提交|取消)(?:已生成|已创建|已提交|成功|完成)"
)
_CLAUSE_SPLIT = re.compile(r"[，。！？；,\.!?;：\n]")
NEGATION = re.compile(r"[没未不无莫勿]|无法|失败|尚未|还没|并未|如果|假如|若您|若是")

CLAIM_NOTE_V1 = (
    "（系统检查：你在本次处理中没有调用任何办理类工具，不能对用户说已经提交、生成确认单、修改或取消。"
    "如果用户已经明确要求办理，请先调用对应工具；如果你说的是之前已经办完的事，或者其实只是查询，请如实修改回复。）"
)
CLAIM_NOTE_V2 = (
    "（系统检查：你这一轮没有调用任何办理类工具，回复里却声称已经提交、生成确认单、修改或取消。请按情况处理："
    "如果用户这一轮要求办理，先调用对应工具（如 prepare_refund / change_address），再根据结果回复；"
    "如果说的是之前已经办好的事，并且这一轮的查询结果能证实（比如订单状态、操作记录），就保持原来的说法，"
    "不要否认之前的操作——之前轮次的工具调用不会出现在你的上下文里；"
    "如果查询结果不能证实，就如实说明当前查到的状态。）"
)
CLAIM_NOTES = (CLAIM_NOTE_V1, CLAIM_NOTE_V2)


def _pick_note() -> str:
    from .config import settings

    return CLAIM_NOTE_V2 if settings.claim_note_version == 2 else CLAIM_NOTE_V1


CLAIM_NOTE = _pick_note()


def _clause_prefix(text: str, pos: int) -> str:
    """取匹配位置所在分句的前缀，用于检测同一分句内的否定或条件假设词。"""
    left = text[:pos]
    parts = _CLAUSE_SPLIT.split(left)
    return parts[-1] if parts else left[max(0, pos - 8):pos]


def claims_done(text: str) -> bool:
    """回复里是否包含肯定语气的“已经办好 / 已生成确认单”表述。"""
    for m in CLAIM.finditer(text):
        prefix = _clause_prefix(text, m.start())
        if not NEGATION.search(prefix):
            return True
    return False


def unbacked_claim(answer: str, calls: list[dict]) -> bool:
    """声称办好了，但本次处理没有调用任何写工具。"""
    return claims_done(answer) and not any(c.get("tool") in WRITE_TOOLS for c in calls)
