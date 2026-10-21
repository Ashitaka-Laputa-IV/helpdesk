"""运行时数据文件的位置。

所有 SQLite 检查点统一放在项目根目录的 .db/ 下，避免散落在仓库根。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DB_DIR = ROOT / ".db"


def db_path(name: str) -> Path:
    """返回 .db/<name>，并确保目录存在。"""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return DB_DIR / name
