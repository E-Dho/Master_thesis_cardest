"""PostgreSQL session and label backend (no numpy needed on the cluster for generation)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


class SqlSession:
    """Thin psycopg wrapper with parameter support and fixed session settings."""

    def __init__(self, host=None, port=None, dbname=None, user=None, statement_timeout_ms: int = 0):
        try:
            import psycopg  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError("the SQL backend requires psycopg") from exc
        self._psycopg = psycopg
        self._kwargs = {
            key: value
            for key, value in {"host": host, "port": port, "dbname": dbname, "user": user}.items()
            if value is not None
        }
        self._statement_timeout_ms = int(statement_timeout_ms)
        self._conn = None

    def __enter__(self) -> "SqlSession":
        self._conn = self._psycopg.connect(**self._kwargs)
        self._conn.autocommit = True
        # v3 compares stored timestamp-without-time-zone values only, but fixing
        # the session time zone keeps any accidental timestamptz use deterministic.
        self.execute("SET TimeZone = 'UTC'")
        self.execute(f"SET statement_timeout = {self._statement_timeout_ms}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._conn is not None:
            self._conn.close()

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> None:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(sql, params)

    def scalar(self, sql: str, params: Optional[Sequence[Any]] = None) -> Any:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return None if row is None else row[0]

    def rows(self, sql: str, params: Optional[Sequence[Any]] = None) -> List[Tuple[Any, ...]]:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [tuple(row) for row in cur.fetchall()]

    @staticmethod
    def is_timeout(error: BaseException) -> bool:
        return type(error).__name__ == "QueryCanceled"


class SqlBackend:
    def __init__(self, session: SqlSession):
        self.session = session

    def count(self, record: Dict[str, Any]) -> int:
        return int(self.session.scalar(record["sql"]))
