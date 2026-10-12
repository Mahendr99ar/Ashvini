"""Settings, read from ASHVINI_* environment variables or a .env file."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASHVINI_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./ashvini.db"

    # where orbits come from
    data_source: Literal["celestrak", "synthetic"] = "synthetic"
    celestrak_groups: str = "active,cosmos-2251-debris,iridium-33-debris,fengyun-1c-debris,cosmos-1408-debris"
    primary_groups: str = "stations"
    primary_norad_ids: str = ""
    synthetic_seed: int = 7
    synthetic_size: int = 2500

    # screening
    screen_days: float = 3.0
    screen_step_s: float = 30.0
    report_miss_km: float = 5.0
    hard_body_radius_m: float = 20.0

    # default decision lines; each organisation can override them
    action_pc: float = 1e-4
    monitor_pc: float = 1e-6

    # optional trained risk model (ONNX, from ml/collision)
    risk_model_path: str = ""

    # space weather
    spaceweather_source: Literal["noaa", "synthetic"] = "synthetic"

    # API
    cors_origins: str = "http://localhost:5173"
    secret: str = "change-me"
    public_read: bool = True

    # alerts by email are optional
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "ashvini@localhost"

    # copilot: any OpenAI-compatible endpoint; without one, the offline writer answers
    copilot_base_url: str = ""
    copilot_api_key: str = ""
    copilot_model: str = ""
    copilot_per_minute: int = 20

    def id_list(self, raw: str) -> list[int]:
        return [int(x) for x in raw.replace(" ", "").split(",") if x]

    def group_list(self, raw: str) -> list[str]:
        return [x.strip() for x in raw.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
