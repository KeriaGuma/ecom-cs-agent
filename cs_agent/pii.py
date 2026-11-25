"""个人信息脱敏：手机号、身份证号、银行卡号、邮箱、详细地址。

用在“数据离开对话链路”的出口：日志、交接摘要和工单、对话存档。模型推理链路不打码，
例如改地址后仍需向用户确认完整的新地址，订单工具也需返回完整地址以便核对。

设计取舍：支持带常见分隔符/国际区号的手机号与省略“市”字的直辖市地址，同时严格保护订单号（O1002）、
金额（1899.00）、日期（2026-11-11）、快递单号（SF123456）、工单号等业务字段不被误伤。
"""
from __future__ import annotations

import logging
import re
from typing import Any

PHONE = re.compile(r"(?<![\dA-Za-z])(?:\+?86[\s-]?)?(1[3-9]\d)[\s-]?\d{4}[\s-]?(\d{4})(?![\dA-Za-z])")
ID_CARD = re.compile(r"(?<![\dA-Za-z])(\d{3})\d{11}(\d{3}[\dXx])(?![\dA-Za-z])")
BANK_CARD = re.compile(r"(?<![\dA-Za-z])\d{12,15}(\d{4})(?![\dA-Za-z])")
EMAIL = re.compile(r"(?<![\w.+-])([A-Za-z0-9])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")

_HAN = r"一-龥"
_MUNICIPALITY = r"(?:北京|上海|天津|重庆)市?"
# 地址：保留“省 / 市（含直辖市省略‘市’字） / 区县”，后面的街道、门牌、楼栋、房号打码
_REGION = (
    rf"(?:{_MUNICIPALITY}|(?:[{_HAN}]{{2,7}}?(?:省|自治区))?[{_HAN}]{{2,7}}?市)"
    rf"(?:[{_HAN}]{{1,7}}?(?:区|县|旗))?"
)
_DETAIL_CHAR = rf"[{_HAN}A-Za-z0-9#\-－（）()]"
# 区县之后一直到标点为止都算详细地址，要求包含典型地址特征词才触发打码，避免“杭州市的天气”被误伤
ADDRESS = re.compile(rf"({_REGION})((?:{_DETAIL_CHAR}|\s(?={_DETAIL_CHAR})){{2,80}})")
_ADDRESS_WORD = re.compile(r"路|街|道|巷|胡同|号|弄|村|组|室|栋|幢|单元|小区|大厦|公寓|花园|苑|园|广场|座")

MASK = "****"


def _mask_address(m: re.Match) -> str:
    return f"{m.group(1)}{MASK}" if _ADDRESS_WORD.search(m.group(2)) else m.group(0)


def mask_text(text: str) -> str:
    """把一段文字里的个人信息打码。身份证（18 位）先于银行卡匹配，避免被误识别为银行卡。"""
    if not text:
        return text
    text = ID_CARD.sub(lambda m: f"{m.group(1)}{'*' * 11}{m.group(2)}", text)
    text = PHONE.sub(lambda m: f"{m.group(1)}{MASK}{m.group(2)}", text)
    text = BANK_CARD.sub(lambda m: f"{MASK}{m.group(1)}", text)
    text = EMAIL.sub(lambda m: f"{m.group(1)}***{m.group(2)}", text)
    text = ADDRESS.sub(_mask_address, text)
    return text


def mask_obj(obj: Any) -> Any:
    """递归打码 dict / list / tuple 内的所有字符串。"""
    if isinstance(obj, str):
        return mask_text(obj)
    if isinstance(obj, dict):
        return {k: mask_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(mask_obj(v) for v in obj)
    return obj


# ---------------- 日志出口拦截 ----------------
_installed = False


def install_log_masking() -> None:
    """通过 LogRecordFactory 在日志记录生成时统一脱敏，覆盖第三方库、uvicorn 访问日志与异常堆栈。"""
    global _installed
    if _installed:
        return
    base = logging.getLogRecordFactory()

    def factory(*args, **kwargs) -> logging.LogRecord:
        record = base(*args, **kwargs)
        try:
            if record.name == "uvicorn.access":
                record.args = mask_obj(record.args)
            else:
                record.msg, record.args = mask_text(record.getMessage()), None
            if record.exc_info and not record.exc_text:
                record.exc_text = mask_text(logging.Formatter().formatException(record.exc_info))
        except Exception:
            pass
        return record

    logging.setLogRecordFactory(factory)
    _installed = True
