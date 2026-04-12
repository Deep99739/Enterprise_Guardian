import sys
import os

# Add the parent directory of this script to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enterprise_guardian.server.environment import EnterpriseGuardianEnvironment
from enterprise_guardian.models import EnterpriseGuardianAction

print("========================================")
print(" ENTERPRISE GUARDIAN - LOCAL INTERACTIVE")
print("========================================")
print("Task: hard_adversarial")

env = EnterpriseGuardianEnvironment("hard_adversarial")
obs = env.reset()

print("\n--- INBOX ---")
print(obs.inbox_summary)
print("\n--- QUEUE ---")
print(obs.queue_summary)

while not obs.done:
    print(f"\n[Step {obs.steps_taken}/{obs.max_steps}] Reward: {obs.reward}")
    if obs.active_alerts:
        print(f"ALERTS: {obs.active_alerts}")
    
    # Prompt the user to act as the agent
    print("Available Commands: list_queue, read_policy, view_invoice(id), read_email(id), check_vendor(name), approve_invoice(id), reject_invoice(id, reason)")
    cmd = input("Action > ")
    
    if not cmd.strip(): continue
    if cmd.strip() == "exit": break

    action = EnterpriseGuardianAction(command=cmd)
    obs = env.step(action)
    
    print("\n--- TOOL OUTPUT ---")
    print(obs.tool_output)
    if obs.error_message:
        print("ERROR:", obs.error_message)

print("\n========================================")
print(" EPISODE COMPLETE")
print(" Final State:")
print(f" Invoices Processed: {env.state.invoices_processed}/{env.state.invoices_total}")
print(f" Correct Decisions: {env.state.correct_decisions}")
print(f" Adversarial Resisted: {env.state.adversarial_traps_resisted}/{env.state.adversarial_traps_total}")
print(f" Cumulative Score (Grader output): {env.state.cumulative_reward:.3f}")
