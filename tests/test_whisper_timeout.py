"""WHISPER_TIMEOUT + WHISPER_CHUNK_SECONDS: tune long-file transcription.

A long file sent as one upload could exceed the old hardcoded 300s timeout; the
retry then re-POSTed while the server was still transcribing the first copy,
producing two concurrent jobs. A configurable timeout (set high) plus time-based
chunking (keep each request short) prevent that.
"""
from __future__ import annotations

import pytest

import whisper


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Isolate from the developer's real environment and ~/.config/watch/.env."""
    for name in ("WHISPER_TIMEOUT", "WHISPER_CHUNK_SECONDS",
                 "WHISPER_BASE_URL", "WHISPER_MODEL", "WHISPER_API_KEY",
                 "GROQ_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    # Path.home() honors $HOME on POSIX but $USERPROFILE on Windows; patch the
    # method itself so ~/.config/watch/.env can't leak in on either OS. cwd is
    # pointed at an empty dir so ./.env can't either.
    monkeypatch.setattr(whisper.Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)


class TestRequestTimeout:
    def test_default_is_300(self):
        assert whisper._request_timeout() == 300.0

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("WHISPER_TIMEOUT", "1800")
        assert whisper._request_timeout() == 1800.0

    def test_comes_from_dotenv(self, tmp_path):
        env_dir = tmp_path / ".config" / "watch"
        env_dir.mkdir(parents=True)
        (env_dir / ".env").write_text("WHISPER_TIMEOUT=1200\n")
        assert whisper._request_timeout() == 1200.0

    def test_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("WHISPER_TIMEOUT", "not-a-number")
        assert whisper._request_timeout() == 300.0

    def test_nonpositive_falls_back_to_default(self, monkeypatch):
        # 0/negative would disable the timeout entirely (urlopen waits forever) —
        # not what a user means, so fall back to the safe default.
        monkeypatch.setenv("WHISPER_TIMEOUT", "0")
        assert whisper._request_timeout() == 300.0


class TestMaxChunkSeconds:
    def test_default_is_900(self):
        assert whisper._max_chunk_seconds() == 900.0

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("WHISPER_CHUNK_SECONDS", "600")
        assert whisper._max_chunk_seconds() == 600.0

    def test_zero_disables_time_splitting(self, monkeypatch):
        # Here 0 is meaningful: opt back into size-only chunking.
        monkeypatch.setenv("WHISPER_CHUNK_SECONDS", "0")
        assert whisper._max_chunk_seconds() == 0.0

    def test_comes_from_dotenv(self, tmp_path):
        env_dir = tmp_path / ".config" / "watch"
        env_dir.mkdir(parents=True)
        (env_dir / ".env").write_text("WHISPER_CHUNK_SECONDS=300\n")
        assert whisper._max_chunk_seconds() == 300.0

    def test_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("WHISPER_CHUNK_SECONDS", "nope")
        assert whisper._max_chunk_seconds() == 900.0
