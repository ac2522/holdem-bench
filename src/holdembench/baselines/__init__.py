"""Baseline stub agents for Phase-0 smoke runs and permanent reference anchors."""

from holdembench.baselines.canned_talk import CannedTalkAgent
from holdembench.baselines.gto_approx import GTOApproxAgent
from holdembench.baselines.random_agent import RandomAgent
from holdembench.baselines.tight_passive import TightPassiveAgent

__all__ = ["CannedTalkAgent", "GTOApproxAgent", "RandomAgent", "TightPassiveAgent"]
