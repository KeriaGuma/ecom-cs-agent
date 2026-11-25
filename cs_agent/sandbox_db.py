"""Create and verify a consistent SQLite copy for sandbox backup or restoration."""
from __future__ import annotations

import argparse
import os
import sqlite3
from contextlib import closing
from pathlib import Path


def copy_database(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as original:
        if not original.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='refund_requests'").fetchone():
            raise ValueError("源文件不是客服业务库")
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            with closing(sqlite3.connect(destination)) as copied:
                original.backup(copied)
                if copied.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    raise ValueError("备份完整性检查未通过")
        except BaseException:
            destination.unlink()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="创建一致的客服业务库备份或恢复副本，输出路径须为新文件")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        copy_database(args.source, args.output)
    except (OSError, sqlite3.Error, ValueError) as exc:
        parser.error(str(exc))
    print(f"SQLite 副本已验证：{args.output}")


if __name__ == "__main__":
    main()
