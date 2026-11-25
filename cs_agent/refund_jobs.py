"""定期按持久化模拟支付账本核对已提交的退款。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import sleep

from .config import settings
from .refunds import RefundService
from .store import ShopStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="模拟支付所用的客服数据库")
    parser.add_argument("--interval", type=float, help="核对间隔秒数，省略时仅运行一次")
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error("客服数据库不存在，请先启动或初始化客服应用")
    if args.interval is not None and (not math.isfinite(args.interval) or args.interval <= 0):
        parser.error("核对间隔必须是大于 0 的有限数值")
    refunds = RefundService(ShopStore(str(args.db)), settings.refund_auto_approve_limit)
    try:
        while True:
            print(json.dumps(refunds.recover_simulated_payments()), flush=True)
            if args.interval is None:
                return
            sleep(args.interval)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
