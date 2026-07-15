"""Env-driven settings. Loaded once via lru_cache."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # Inbound auth: accepted bearer tokens (comma-separated in env, or a JSON array).
    relay_api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # Upstream OpenAI-compatible provider.
    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_api_key: str = ""
    upstream_organization: str = ""
    upstream_timeout: float = 600.0

    # Model fallback when a request omits `model`.
    default_model: str = ""

    # Server.
    listen_host: str = "127.0.0.1"
    listen_port: int = 8787
    log_level: str = "info"

    # ── P1 router ──
    # Master switch. When off, the relay is pure passthrough (P0 behavior).
    router_enabled: bool = False
    # Run + record decisions but do NOT override the client model. Safe rollout
    # mode to validate the rule scorer before trusting it (like OpenSquilla's
    # rollout_phase="observe").
    router_observe_only: bool = False
    # Hard budget for scoring (seconds). Timeout → passthrough, never block.
    router_timeout_seconds: float = 2.0
    # Tier → {model, description}. JSON env. Empty → built-in DEFAULT_TIERS.
    router_tiers: Annotated[dict, NoDecode] = Field(default_factory=dict)
    # Sticky window: how many recent tiers to keep per session for anti-flapping.
    router_sticky_turns: int = 3
    # Margin below this → confidence_gate upgrades one tier.
    router_confidence_threshold: float = 0.55
    # Context length (chars, ≈4×tokens) above which large_context_floor → c2.
    router_large_context_chars: int = 64000
    # Optional SQLite path for decision records. Empty = in-memory only.
    router_decision_db: str = ""
    router_log_decisions: bool = True
    # P3 prep: directory for date-partitioned JSONL capture
    # (router-samples-YYYY-MM-DD.jsonl). Empty = no capture. This is the training
    # data source for a future P3 LightGBM head; also useful for audit/cost.
    router_capture_dir: str = ""
    # When true, also write the last user message text to a sidecar file
    # (router-raw-YYYY-MM-DD.jsonl) for LLM-as-judge offline labeling.
    # Default false — privacy-preserving (no prompt text stored unless opted in).
    capture_raw_content: bool = False
    # P3: path to a trained LightGBM model (output of scripts/train_p3.py).
    # When set, the ML head replaces the rule scorer at the apply_router seam.
    # Empty = use the rule scorer (default). If the model fails to load, the
    # caller transparently falls back to the rule scorer (never blocks).
    router_ml_model_path: str = ""

    # ── P2 ensemble (B5 fusion: N proposers → 1 aggregator) ──
    # Ensemble wraps AFTER routing; only fires for routed tiers >= ensemble_min_tier
    # so easy turns stay single-model (cost/latency). Requires ROUTER_ENABLED=true.
    ensemble_enabled: bool = False
    # Ensemble mode: "b5_fusion" (aggregator fuses drafts) or "best_of_n" (scorer
    # picks the single best draft). B5 is better for synthesis; best_of_n is
    # cheaper and better for tasks with a clear correct answer.
    ensemble_mode: str = "b5_fusion"
    # Proposer model ids (comma-separated in env). The routed anchor model is
    # auto-prepended if not already present.
    ensemble_proposers: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Aggregator model id; empty = use the routed anchor as aggregator too.
    ensemble_aggregator: str = ""
    # Scorer model for best_of_n mode; empty = use the aggregator.
    ensemble_scorer_model: str = ""
    # Only fuse when the routed tier rank is >= this (c0<c1<c2<c3).
    ensemble_min_tier: str = "c2"
    # Quorum: minimum successful proposers before aggregation.
    ensemble_min_successful: int = 2
    # Cap on the number of proposers (cost control; anchor counts toward the cap).
    ensemble_max_proposers: int = 3
    ensemble_proposer_timeout: float = 60.0
    ensemble_aggregator_timeout: float = 120.0
    # Truncate each candidate draft to this many chars in the aggregator prompt.
    ensemble_candidate_max_chars: int = 24000

    @field_validator("relay_api_keys", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> list[str]:
        # Accept either a comma-separated string or a JSON array from env.
        if isinstance(value, str):
            return [token.strip() for token in value.split(",") if token.strip()]
        return list(value) if value else []

    @field_validator("router_tiers", mode="before")
    @classmethod
    def _parse_router_tiers(cls, value: object) -> dict:
        # Accept a JSON object string (NoDecode keeps the raw env string) or a dict.
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            return json.loads(text)  # raises on malformed JSON — fail loud
        return value or {}

    @field_validator("ensemble_proposers", mode="before")
    @classmethod
    def _split_proposers(cls, value: object) -> list[str]:
        # Model ids have no commas, so CSV is ergonomic. Also accepts a JSON list.
        if isinstance(value, str):
            return [m.strip() for m in value.split(",") if m.strip()]
        return list(value) if value else []


@lru_cache
def get_settings() -> Settings:
    return Settings()
