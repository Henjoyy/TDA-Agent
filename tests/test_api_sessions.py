"""
API 세션 수명주기(TTL/LRU) 테스트
"""
from __future__ import annotations

from datetime import timedelta

import api.main as api_main


class _Dummy:
    pass


def test_expired_session_is_purged(monkeypatch):
    with api_main._sessions_lock:
        api_main._sessions.clear()

    monkeypatch.setattr(api_main, "SESSION_TTL", timedelta(seconds=1))

    old_now = api_main._utcnow() - timedelta(seconds=10)
    with api_main._sessions_lock:
        api_main._sessions["old"] = api_main.SessionEntry(
            pipeline=_Dummy(),
            result=_Dummy(),
            created_at=old_now,
            last_accessed=old_now,
        )

    assert api_main._get_session("old") is None


def test_lru_limit_removes_oldest_session(monkeypatch):
    with api_main._sessions_lock:
        api_main._sessions.clear()

    monkeypatch.setattr(api_main, "MAX_SESSIONS", 2)

    api_main._set_session("s1", _Dummy(), _Dummy())
    api_main._set_session("s2", _Dummy(), _Dummy())
    api_main._set_session("s3", _Dummy(), _Dummy())

    with api_main._sessions_lock:
        assert "s1" not in api_main._sessions
        assert "s2" in api_main._sessions
        assert "s3" in api_main._sessions
