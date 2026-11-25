"""轻量级售后政策关键词检索（零外部依赖）。

按二级标题切分条款，使用字符二元组重叠度计算匹配分：
1. 在离线单元测试和无 API Key 沙箱演示中作为轻量检索实现；
2. 在主系统外部模型/检索服务故障时，作为零外部依赖的本地 FAQ 降级兜底。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Section:
    source: str
    title: str
    text: str


class PolicyKB:
    def __init__(self, kb_dir: str):
        self.sections: list[Section] = []
        for path in sorted(Path(kb_dir).glob("*.md")):
            self.sections.extend(_split(path))

    def search(self, query: str, k: int = 3) -> list[Section]:
        q = _bigrams(query)
        if not q:
            return []
        scored = [(len(q & _bigrams(s.title + s.text)), s) for s in self.sections]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]


    def best_match(self, query: str) -> Optional[tuple[float, Section]]:
        """最相关的一条及其匹配度（问题里的字符二元组有多大比例出现在该条款中，0～1）。FAQ 降级用。"""
        q = _bigrams(query)
        if not q:
            return None
        score, sec = max(((len(q & _bigrams(s.title + s.text)) / len(q), s) for s in self.sections),
                         key=lambda x: x[0], default=(0.0, None))
        return (score, sec) if sec is not None and score > 0 else None


def _split(path: Path) -> list[Section]:
    sections, title, buf = [], None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if title:
                sections.append(Section(path.name, title, "\n".join(buf).strip()))
            title, buf = line[3:].strip(), []
        elif title:
            buf.append(line)
    if title:
        sections.append(Section(path.name, title, "\n".join(buf).strip()))
    return sections


def _bigrams(text: str) -> set[str]:
    chars = [c for c in text if not c.isspace() and c not in "，。、？！：；,.?!:;（）()"]
    return {a + b for a, b in zip(chars, chars[1:])}
