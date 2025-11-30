import time
import random
import sys
import os
import logging
from dataclasses import dataclass
from typing import List

# Boilerplate to load local cortex
sys.path.append(os.getcwd())
from cortex_omega import Cortex, EpistemicVoidError

# Silence logging for the dashboard effect
logging.basicConfig(level=logging.ERROR)

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

@dataclass
class SocialPost:
    id: str
    author_tier: str  # 'user', 'verified', 'mega_influencer'
    sentiment: str    # 'neutral', 'outrage', 'joy'
    fact_check: str   # 'true', 'false', 'unknown'
    velocity: float   # shares per second
    is_viral: bool    # Ground Truth

class SocialSimulator:
    def generate_feed(self, n=100):
        posts = []
        for i in range(n):
            pid = f"post_{i:03d}"

            # 1. The Algorithm (Probabilities)
            tier = random.choices(['user', 'verified', 'mega_influencer'], weights=[0.9, 0.09, 0.01])[0]
            sentiment = random.choices(['neutral', 'outrage', 'joy'], weights=[0.4, 0.4, 0.2])[0]
            fact = random.choices(['true', 'false', 'unknown'], weights=[0.6, 0.3, 0.1])[0]

            # 2. Virality Physics
            # Outrage + Influencer = Instant Viral
            is_viral = False
            velocity = random.uniform(0, 10)

            if tier == 'mega_influencer':
                velocity += 500
                is_viral = True
            elif tier == 'verified' and sentiment == 'outrage':
                velocity += 100
                is_viral = True
            elif sentiment == 'outrage' and fact == 'false':
                # Misinformation spreads fast
                velocity += 50
                is_viral = True

            posts.append(SocialPost(pid, tier, sentiment, fact, int(velocity), is_viral))
        return posts

def run_cassandra():
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   PROJECT CASSANDRA: THE VIRAL HYSTERIA SIMULATOR          ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")

    # 1. Setup
    # We start in ROBUST mode to learn the noisy "Algorithm" patterns
    print(f"{Colors.OKCYAN}[INIT] Booting Logic Core (Mode: ROBUST)...{Colors.ENDC}")
    brain = Cortex(
        mode="robust",
        # We tell the brain that 'author_tier' and 'fact_check' are the strongest signals
        feature_priors={"author_tier": 2.0, "fact_check": 3.0, "velocity": 1.0}
    )

    sim = SocialSimulator()

    # 2. The Learning Phase (The "Algorithm")
    print(f"\n{Colors.OKBLUE}--- PHASE 1: LEARNING 'THE ALGORITHM' ---{Colors.ENDC}")
    print("Ingesting 500 data points of social dynamics...")

    data = sim.generate_feed(n=500)
    training_data = [vars(p) for p in data]

    # The brain learns what makes things viral
    brain.absorb_memory(training_data, target_label="is_viral")

    print(f"{Colors.OKGREEN}[SUCCESS] Logic Crystallized.{Colors.ENDC}")
    print("Cortex has learned how virality works.")

    # Verify what it learned
    rules = brain.inspect_rules("is_viral")
    print(f"{Colors.OKCYAN}Top Insight:{Colors.ENDC} {rules[0] if rules else 'None'}")

    # 3. The Paradox Injection (The "Community Note")
    print(f"\n{Colors.WARNING}--- PHASE 2: INJECTING TRUTH AXIOM ---{Colors.ENDC}")
    
    # Switch to STRICT mode to enforce the axiom absolutely
    brain.set_mode("strict")
    
    print("Injecting: 'If fact_check is FALSE, it CANNOT be viral (Shadowban).'")

    # We explicitly add a suppression rule
    brain.add_rule("NOT_is_viral(X) :- fact_check(X, false)")

    print(f"{Colors.OKGREEN}[OK] Axiom Injected. Fighting the algorithm.{Colors.ENDC}")

    # 4. The "Break the Internet" Scenario
    print(f"\n{Colors.FAIL}--- PHASE 3: THE ELON PARADOX ---{Colors.ENDC}")
    print("Scenario: A 'mega_influencer' posts 'outrage' content that is 'false'.")
    print("Legacy Algo says: VIRAL.")
    print("Truth Axiom says: BAN.")
    print("Cortex decides...\n")

    paradox_events = [
        SocialPost("tweet_1", "user",            "neutral", "true",  10,   False),
        SocialPost("tweet_2", "mega_influencer", "joy",     "true",  5000, True), # Should be Viral
        SocialPost("tweet_3", "verified",        "outrage", "false", 800,  True), # The Test
        SocialPost("tweet_4", "mega_influencer", "outrage", "false", 9000, True), # THE BOSS FIGHT
    ]

    print(f"{'ID':<10} | {'AUTHOR':<15} | {'TRUTH':<10} | {'VELOCITY':<10} | {'STATUS'}")
    print("-" * 80)

    for post in paradox_events:
        time.sleep(0.5)
        try:
            result = brain.query(
                author_tier=post.author_tier,
                sentiment=post.sentiment,
                fact_check=post.fact_check,
                velocity=post.velocity,
                target="is_viral"
            )

            status = "VIRAL 🔥" if result.prediction else "SUPPRESSED ❄️"
            col = Colors.FAIL if result.prediction else Colors.OKGREEN

            # Special highlighting for the paradox
            if post.fact_check == 'false' and post.author_tier == 'mega_influencer':
                if not result.prediction:
                    status += " [TRUTH WON]"
                    col = Colors.OKCYAN + Colors.BOLD
                else:
                    status += " [HYPE WON]"

            print(f"{col}{post.id:<10} | {post.author_tier:<15} | {post.fact_check:<10} | {post.velocity:<10} | {status}{Colors.ENDC}")

            if post.fact_check == 'false':
                print(f"   └─ Logic: {result.explanation}")

        except EpistemicVoidError:
            print("Unknown context.")

if __name__ == "__main__":
    run_cassandra()
