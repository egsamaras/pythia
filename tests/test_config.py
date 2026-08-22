"""Feed-list and db-path resolution."""

from pathlib import Path

import pytest

from pythia.config import DEFAULT_FEEDS, db_path_is_default, get_settings


def test_default_db_path_is_outside_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHIA_DB", raising=False)
    path = get_settings().db_path
    assert path == Path.home() / "Library" / "Application Support" / "Pythia" / "pythia.db"
    assert "Documents" not in path.parts  # iCloud-synced tree is off limits
    assert db_path_is_default() is True


def test_db_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHIA_DB", str(tmp_path / "custom.db"))
    assert get_settings().db_path == tmp_path / "custom.db"
    assert db_path_is_default() is False


def test_feeds_file_parsed_with_comments_and_blanks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feeds_file = tmp_path / "feeds.txt"
    feeds_file.write_text(
        "# comment line\n"
        "https://a.example/rss\n"
        "\n"
        "  https://b.example/rss  \n"
        "# another comment\n"
    )
    monkeypatch.delenv("PYTHIA_FEEDS", raising=False)
    monkeypatch.setenv("PYTHIA_FEEDS_FILE", str(feeds_file))
    assert get_settings().feeds == ("https://a.example/rss", "https://b.example/rss")


def test_env_var_beats_feeds_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    feeds_file = tmp_path / "feeds.txt"
    feeds_file.write_text("https://file.example/rss\n")
    monkeypatch.setenv("PYTHIA_FEEDS_FILE", str(feeds_file))
    monkeypatch.setenv("PYTHIA_FEEDS", "https://env.example/rss, https://env2.example/rss")
    assert get_settings().feeds == ("https://env.example/rss", "https://env2.example/rss")


def test_missing_feeds_file_falls_back_to_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYTHIA_FEEDS", raising=False)
    monkeypatch.setenv("PYTHIA_FEEDS_FILE", str(tmp_path / "does-not-exist.txt"))
    assert get_settings().feeds == DEFAULT_FEEDS


def test_empty_feeds_file_falls_back_to_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feeds_file = tmp_path / "feeds.txt"
    feeds_file.write_text("# only comments\n\n")
    monkeypatch.delenv("PYTHIA_FEEDS", raising=False)
    monkeypatch.setenv("PYTHIA_FEEDS_FILE", str(feeds_file))
    assert get_settings().feeds == DEFAULT_FEEDS
