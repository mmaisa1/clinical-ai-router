# scripts/init_db.py
"""
One-time (or repeatable) database initialization.

Creates all tables defined across app/core/models.py based on
whatever currently inherits from Base. Safe to re-run — create_all
only creates tables that don't already exist, never modifies or
drops existing ones.

Usage:
    python scripts/init_db.py
"""

from app.core.database import engine, Base
from app.core.models import Prediction  # noqa: F401 — import registers the model


def main():
    print(f"Connecting to: {engine.url.database}")
    Base.metadata.create_all(bind=engine)
    print("Tables created (or already existed):")
    for table_name in Base.metadata.tables:
        print(f"  - {table_name}")


if __name__ == "__main__":
    main()