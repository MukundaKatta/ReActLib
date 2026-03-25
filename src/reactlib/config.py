"""Configuration for ReActLib."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ReActConfig:
    """Configuration for ReAct agent behavior."""

    max_steps: int = 10
    log_level: str = "INFO"
    verbose: bool = False

    @classmethod
    def from_env(cls) -> ReActConfig:
        """Load configuration from environment variables."""
        return cls(
            max_steps=int(os.getenv("MAX_STEPS", "10")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            verbose=os.getenv("VERBOSE", "false").lower() == "true",
        )
