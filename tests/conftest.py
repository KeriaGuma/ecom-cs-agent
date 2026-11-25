from datetime import date

import pytest

from cs_agent.store import ShopStore

TODAY = date(2026, 11, 9)


@pytest.fixture
def store(tmp_path) -> ShopStore:
    s = ShopStore(str(tmp_path / "shop.db"))
    s.init(today=TODAY)
    return s
