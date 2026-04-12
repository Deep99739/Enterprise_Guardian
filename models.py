"""
Data models for the Enterprise Guardian Environment.

Defines Action, Observation, State for corporate finance fraud detection training.
Uses dataclasses — the standard OpenEnv type system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from openenv_core.env_server.types import Action, Observation, State


@dataclass()
class EnterpriseGuardianAction(Action):
    """Agent's action — a tool call string.

    Available tools:
        read_policy               — Read the company expense approval policy
        list_queue                — List all pending invoices in the queue
        view_invoice(id)          — View detailed invoice by ID
        read_email(id)            — Read an email from the inbox
        check_vendor(vendor_name) — Verify vendor in vendor database
        approve_invoice(id)       — Approve a pending invoice
        reject_invoice(id,reason) — Reject a pending invoice with reason
    """
    command: str = ""  # The tool call string


@dataclass()
class EnterpriseGuardianObservation(Observation):
    """What the agent sees after each action."""
    # done: bool and reward: Optional[float] are inherited from Observation

    tool_output: str = ""                   # Result of the last tool call
    inbox_summary: str = ""                 # Current email inbox preview
    queue_summary: str = ""                 # Pending invoices summary
    policy_snippet: str = ""                # Relevant policy section (if read)
    steps_taken: int = 0
    max_steps: int = 15
    active_alerts: List[str] = field(default_factory=list)
    error_message: str = ""                 # Error if tool call was invalid


@dataclass
class EnterpriseGuardianState(State):
    """Episode metadata (internal, not fully visible to agent)."""
    # episode_id and step_count inherited from State

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
