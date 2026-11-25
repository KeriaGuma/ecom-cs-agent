"""个人信息脱敏：覆盖分隔符手机号、直辖市省略“市”字地址，以及日志、工单、对话存档三个出口。"""
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from uvicorn.logging import AccessFormatter

from cs_agent.handoff import create_ticket
from cs_agent.memory import make_memory_node
from cs_agent.pii import install_log_masking, mask_obj, mask_text

MASKED = [
    ("我手机13812345678，快点", "我手机138****5678，快点"),
    ("电话 138-1234-5678 联系我", "电话 138****5678 联系我"),
    ("手机 +86 139 8765 4321", "手机 139****4321"),
    ("身份证33010619900101123X", "身份证330***********123X"),
    ("退到银行卡6222021234567890123", "退到银行卡****0123"),
    ("邮箱 keria.guma@gmail.com", "邮箱 k***@gmail.com"),
    ("改成杭州市西湖区文三路 1 号", "改成杭州市西湖区****"),
    ("寄到上海市浦东新区世纪大道100号3栋2单元501室，谢谢", "寄到上海市浦东新区****，谢谢"),
    ("上海浦东新区世纪大道100号", "上海浦东新区****"),
    ("北京朝阳区建国路88号SOHO现代城A座1203室", "北京朝阳区****"),
    ("北京市朝阳区建国路88号SOHO现代城A座1203室", "北京市朝阳区****"),
    ("浙江省杭州市余杭区文一西路969号", "浙江省杭州市余杭区****"),
]

UNTOUCHED = [
    "订单 O1002 退款 1899.00 元", "2026-11-11 到达杭州转运中心，预计明天送达", "顺丰 SF123456，中通 ZT987654",
    "工单号 #12，您是下一位", "支付宝 1–3 个工作日，银行卡 3–7 个工作日", "满 99 元包邮", "签收 5 天，15 天内可退",
    "积分 1260 分，最多抵扣 10%", "杭州市的天气", "我在杭州市，东西还没到", "上海市浦东新区的仓库发货",
]


@pytest.mark.parametrize("raw,expected", MASKED)
def test_masks_personal_info(raw, expected):
    assert mask_text(raw) == expected


@pytest.mark.parametrize("text", UNTOUCHED)
def test_leaves_business_fields_alone(text):
    assert mask_text(text) == text


def test_mask_obj_walks_nested_structures():
    out = mask_obj({"pending": "电话138-1234-5678", "done": ["改到北京朝阳区建国路 88 号"], "n": 3})
    assert out == {"pending": "电话138****5678", "done": ["改到北京朝阳区****"], "n": 3}


def test_logs_and_tracebacks_are_masked(caplog):
    install_log_masking()
    log = logging.getLogger("cs_agent.test")
    with caplog.at_level(logging.WARNING):
        log.warning("用户 %s 要求改到 %s", "13812345678", "杭州市西湖区文三路 1 号")
        try:
            raise ValueError("地址非法：上海市浦东新区世纪大道100号")
        except ValueError:
            log.exception("处理失败")
    text = caplog.text
    assert "13812345678" not in text and "文三路" not in text and "世纪大道" not in text
    assert "138****5678" in text and "上海市浦东新区****" in text


def test_uvicorn_access_formatter_keeps_http_fields_and_masks_query():
    install_log_masking()
    record = logging.getLogger("uvicorn.access").makeRecord(
        "uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "GET", "/path?phone=13812345678", "1.1", 200), None,
    )
    text = AccessFormatter('%(client_addr)s "%(request_line)s" %(status_code)s').format(record)
    assert "138****5678" in text and "13812345678" not in text
    assert "GET /path?" in text and "200 OK" in text


def _state(text):
    return {"user_id": "u001", "messages": [HumanMessage(text)], "turn_results": [], "facts": {}}


def test_handoff_ticket_is_masked_before_it_is_stored(store):
    state = _state("投诉！我电话13812345678，地址杭州市西湖区文三路 1 号，东西一直没到")
    ticket = create_ticket(store, None, state, "用户投诉", use_llm=False, mask=True)
    stored = store.open_tickets()[0]
    for blob in (str(ticket), str(stored)):
        assert "13812345678" not in blob and "文三路" not in blob
    assert ticket["priority"] == 1


def test_handoff_ticket_unmasked_when_switched_off(store):
    ticket = create_ticket(store, None, _state("我电话13812345678"), "用户要求人工", use_llm=False, mask=False)
    assert "13812345678" in str(ticket)


def test_transcript_archive_is_masked(store):
    node = make_memory_node(store, llm=None, mask=True)
    msgs = [HumanMessage("改到杭州市西湖区文三路 1 号，电话13812345678", id="m1"),
            AIMessage("已将订单 O1005 的收货地址修改为：杭州市西湖区文三路 1 号", id="m2")]
    node({"messages": msgs}, {"configurable": {"thread_id": "t1"}})
    archived = " ".join(r["content"] for r in store.transcript("t1"))
    assert "文三路" not in archived and "13812345678" not in archived
    assert "O1005" in archived
