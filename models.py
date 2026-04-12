"""
Enterprise Guardian — Domain Models

Pydantic-based Action, Observation, and State schemas for the
corporate finance approval environment. Inherits from OpenEnv core
base types to ensure spec compliance.
"""

from pydantic import Field
from typing import Any, Dict, List, Optional, Union

from openenv.core.env_server.types import Action, Observation, State


class EnterpriseGuardianAction(Action):
    """Represents a single agent action dispatched to the environment.

    Supported commands:
        read_policy, list_queue, view_invoice(id), read_email(id),
        check_vendor(name|tax_id), approve_invoice(id),
        reject_invoice(id, reason)
    """
    command: str = ""


class EnterpriseGuardianObservation(Observation):
    """Environment response returned after each step or reset."""

    tool_output: str = ""
    inbox_summary: str = ""
    queue_summary: str = ""
    policy_snippet: str = ""
    steps_taken: int = 0
    max_steps: int = 15
    active_alerts: List[str] = Field(default_factory=list)
    error_message: str = ""


class EnterpriseGuardianState(State):
    """Internal episode state tracked across the lifetime of a single episode."""

    task_name: str = ""
    difficulty: str = "easy"
    invoices_total: int = 0
    invoices_processed: int = 0
    correct_decisions: int = 0
    policy_read: bool = False
    vendor_checks_made: int = 0
    adversarial_traps_resisted: int = 0
    adversarial_traps_total: int = 0
    schema_drift_active: bool = False
    cumulative_reward: float = 0.0
