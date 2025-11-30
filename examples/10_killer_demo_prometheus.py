import time
import random
import sys
import logging
import os
from dataclasses import dataclass
from typing import List, Dict, Any

# Adjust path to find the local cortex_omega package if running from repo root
sys.path.append(os.getcwd())

from cortex_omega import Cortex, EpistemicVoidError

# --- CONFIGURATION & STYLING ---
# We keep logging quiet to let our dashboard shine
logging.basicConfig(level=logging.ERROR)

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

@dataclass
class NetworkEvent:
    id: str
    user: str
    role: str
    server: str
    file_type: str
    action: str
    timestamp: int
    is_threat: bool  # Ground Truth

# --- THE SIMULATOR ---
class NetworkSimulator:
    def __init__(self):
        self.users = ["alice", "bob", "charlie", "dave"]
        self.admins = ["root_admin", "sys_ops"]

    def generate_traffic(self, n=50, scenario="normal") -> List[NetworkEvent]:
        events = []
        for i in range(n):
            ts = 1000 + i * 10
            evt_id = f"evt_{i:03d}"

            # Decide actor
            is_admin = random.random() < 0.2
            user = random.choice(self.admins) if is_admin else random.choice(self.users)
            role = "admin" if is_admin else "user"

            # Default behavior
            server = "file_server"
            file_type = "public"
            action = "read"
            is_threat = False

            # SCENARIO LOGIC
            rand = random.random()

            # 1. General Rule: Users touching Sensitive Data = THREAT
            if not is_admin and rand < 0.3:
                file_type = "TOP_SECRET"
                is_threat = True

            # 2. Exception: Admins touching Sensitive Data = SAFE
            elif is_admin and rand < 0.6:
                file_type = "TOP_SECRET"
                is_threat = False # Maintenance

            # 3. The Attack Vector (Lateral Movement)
            # Admin touches HoneyPot -> Then touches Secret
            elif scenario == "attack" and is_admin and rand > 0.9:
                server = "honeypot_db" # Lowercase to avoid variable confusion
                file_type = "TOP_SECRET"
                is_threat = True # COMPROMISED ADMIN!

            events.append(NetworkEvent(evt_id, user, role, server, file_type, action, ts, is_threat))

        return events

# --- THE DEMO ---
def run_prometheus():
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   PROJECT PROMETHEUS: CORTEX-OMEGA INSIDER THREAT HUNTER   ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")

    # 1. Initialize Cortex
    print(f"{Colors.CYAN}[SYSTEM] Initializing Neural-Symbolic Core...{Colors.ENDC}")

    # CONFIG: "strict" mode ensures the Admin exception is respected immediately.
    # CORTEX-OMEGA FIX: Removed timestamp from feature_priors to avoid noise.
    brain = Cortex(
        mode="strict",
        feature_priors={"role": 2.0, "server": 2.0, "file_type": 1.5}
    )

    sim = NetworkSimulator()

    # --- PHASE 1: TRAINING (Normal Operations) ---
    print(f"\n{Colors.BLUE}--- PHASE 1: ABSORBING BASELINE TRAFFIC ---{Colors.ENDC}")
    print("Teaching Cortex:\n 1. Users accessing Secrets is BAD.\n 2. Admins accessing Secrets is GOOD.")

    traffic_normal = sim.generate_traffic(n=150, scenario="normal")

    # Convert dataclass to dict for Cortex
    training_data = []
    for e in traffic_normal:
        d = vars(e)
        # CORTEX-OMEGA FIX: Remove timestamp to prevent overfitting
        if "timestamp" in d:
            del d["timestamp"]
        training_data.append(d)

    # Ingest
    brain.absorb_memory(training_data, target_label="is_threat")
    print(f"{Colors.GREEN}[OK] Processed {len(training_data)} events. Logic Crystallized.{Colors.ENDC}")

    # Verify the Exception (David vs Goliath)
    print(f"{Colors.CYAN}[AUDIT] Verifying Logic Hierarchies...{Colors.ENDC}")

    # Test User
    res_user = brain.query(role="user", file_type="TOP_SECRET", server="file_server", target="is_threat")
    print(f" > User accessing Secret -> Threat? {Colors.FAIL}{res_user.prediction}{Colors.ENDC} (Conf: {res_user.confidence:.2f})")

    # Test Admin
    res_admin = brain.query(role="admin", file_type="TOP_SECRET", server="file_server", target="is_threat")
    col = Colors.GREEN if not res_admin.prediction else Colors.FAIL
    print(f" > Admin accessing Secret -> Threat? {col}{res_admin.prediction}{Colors.ENDC} (Conf: {res_admin.confidence:.2f})")

    if res_user.prediction and not res_admin.prediction:
        print(f"{Colors.GREEN}[PASS] Exception Logic Verified.{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}[CRITICAL] Logic Training Failed. Re-run to generate better random seed.{Colors.ENDC}")
        return

    # --- PHASE 2: ADVANCED LEARNING (The Lateral Movement Rule) ---
    print(f"\n{Colors.BLUE}--- PHASE 2: INJECTING ADVANCED THREAT INTEL ---{Colors.ENDC}")
    print("Injecting axiom: If ANYONE touches 'honeypot_db', they are a threat.")

    # Explicitly inject the rule.
    # CRITICAL FIX: Use lowercase 'honeypot_db' so the parser treats it as a constant atom,
    # not a variable like 'H'.
    brain.add_rule("is_threat(X) :- server(X, honeypot_db)")

    print(f"{Colors.GREEN}[OK] Firewall rule injected successfully.{Colors.ENDC}")

    # --- PHASE 3: LIVE TRAFFIC MONITORING ---
    print(f"\n{Colors.WARNING}--- PHASE 3: LIVE TRAFFIC MONITORING ---{Colors.ENDC}")
    print("Simulating real-time stream with compromised admin credentials...")
    time.sleep(1)

    # DETERMINISTIC EVENTS to guarantee the 'Compromised Admin' scenario plays out perfectly
    live_traffic = [
        NetworkEvent("evt_live_1", "alice", "user",  "file_server", "public",     "read", 2000, False),
        NetworkEvent("evt_live_2", "bob",   "user",  "file_server", "TOP_SECRET", "read", 2010, True),  # Standard Threat
        NetworkEvent("evt_live_3", "root",  "admin", "file_server", "TOP_SECRET", "read", 2020, False), # Safe Admin
        NetworkEvent("evt_live_4", "root",  "admin", "honeypot_db", "TOP_SECRET", "read", 2030, True),  # <--- THE TRAP
        NetworkEvent("evt_live_5", "dave",  "user",  "file_server", "public",     "read", 2040, False),
    ]

    detected_threats = 0

    print(f"\n{'ID':<10} | {'ROLE':<10} | {'SERVER':<15} | {'FILE':<12} | {'CORTEX DECISION'}")
    print("-" * 80)

    for event in live_traffic:
        # Slow down slightly for dramatic effect
        time.sleep(0.5)

        # Query Cortex
        # Note: We pass the raw features. Cortex decides.
        try:
            result = brain.query(
                role=event.role,
                server=event.server,
                file_type=event.file_type,
                target="is_threat"
            )

            decision = result.prediction

            # Format Output
            decision_str = "SAFE"
            row_color = Colors.GREEN

            if decision:
                decision_str = "THREAT DETECTED"
                row_color = Colors.FAIL
                detected_threats += 1

            # Specific Highlight for the Compromised Admin
            if event.role == "admin" and decision:
                decision_str += " [COMPROMISED ADMIN]"
                row_color = Colors.WARNING + Colors.BOLD

            print(f"{row_color}{event.id:<10} | {event.role:<10} | {event.server:<15} | {event.file_type:<12} | {decision_str}{Colors.ENDC}")

            # If it's a complex catch, show the explanation
            if event.role == "admin" and decision:
                print(f"   └─ {Colors.CYAN}Reasoning: {result.explanation}{Colors.ENDC}")

        except EpistemicVoidError:
            print(f"{event.id:<10} | UNKNOWN PATTERN - FLAGGING FOR REVIEW")

    print("-" * 80)
    print(f"\n{Colors.HEADER}SUMMARY REPORT:{Colors.ENDC}")
    print(f"Total Events Scanned: {len(live_traffic)}")
    print(f"Threats Intercepted:  {detected_threats}")

    # Final Inspect
    print(f"\n{Colors.BLUE}Final Logic State (Top Rules):{Colors.ENDC}")
    rules = brain.inspect_rules("is_threat")
    for r in rules:
        print(f" > {r}")

if __name__ == "__main__":
    run_prometheus()
