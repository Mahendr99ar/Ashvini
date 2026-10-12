from datetime import UTC, datetime

import pytest

from ashvini.config import Settings
from ashvini.db import make_sessionmaker
from ashvini.ingest.synthetic import build_world
from ashvini.pipeline import run_pipeline

NOW = datetime(2026, 10, 10, 6, 0, tzinfo=UTC)


@pytest.fixture(scope="session")
def world():
    return build_world(7, 1200, NOW)


def make_settings(tmp_path, **extra) -> Settings:
    return Settings(database_url=f"sqlite:///{tmp_path}/test.db", data_source="synthetic", synthetic_size=1200,
                    screen_days=2.0, spaceweather_source="synthetic", _env_file=None, **extra)


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def factory(settings):
    return make_sessionmaker(settings.database_url)


@pytest.fixture
def ran(settings, factory, tmp_path):
    summary = run_pipeline(settings, factory, NOW, cache_dir=tmp_path / "cache")
    return summary
