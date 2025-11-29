import matplotlib.pyplot as plt
import numpy as np
from cortex_omega import Cortex
import sys
import os

# --- REAL-TIME OBSERVABILITY DEMO ---
# This script demonstrates how to extract internal confidence metrics
# from the Cortex kernel in real-time to build observability dashboards.

def run_live_experiment():
    print("[REAL-TIME] Initializing CORTEX-Ω Kernel...")
    brain = Cortex()
    
    # Metrics Storage
    history_blue = [] # Stable Rule: Blue -> Float
    history_red = []  # Volatile Rule: Red -> Glow
    
    cycles = 50
    kill_switch_cycle = 30
    
    print(f"[REAL-TIME] Starting {cycles} inference cycles...")

    for i in range(cycles):
        # --- PHASE 1: TRAINING (Data Ingestion) ---
        
        # 1. Stable Data Stream (Blue objects always float)
        brain.absorb_memory([
            {'color': 'blue', 'mass': 'light', 'result': True} # Target: float
        ], target_label='float')

        # 2. Volatile Data Stream (Red objects glow... until they don't)
        if i < kill_switch_cycle:
            # Before Kill Switch: Red -> Glow
            brain.absorb_memory([
                {'color': 'red', 'mass': 'heavy', 'result': True} # Target: glow
            ], target_label='glow')
        else:
            # KILL SWITCH ACTIVATED: Context Shift
            # Now Red objects DO NOT glow.
            if i == kill_switch_cycle:
                print(f"[EVENT] !!! CONTEXT SHIFT DETECTED AT CYCLE {i} !!!")
            
            brain.absorb_memory([
                {'color': 'red', 'mass': 'heavy', 'result': False} # Target: glow
            ], target_label='glow')

        # --- PHASE 2: INSPECTION (Extracting Metrics) ---
        
        # Metric 1: Confidence in Stable Rule
        # Query: Blue + Light -> Float?
        q_blue = brain.query(color='blue', mass='light') 
        if q_blue and q_blue.prediction is True:
             conf_blue = q_blue.confidence
        else:
             conf_blue = 0.0
        history_blue.append(conf_blue)

        # Metric 2: Confidence in Volatile Rule
        # Query: Red + Heavy -> Glow?
        q_red = brain.query(color='red', mass='heavy')
        
        if q_red and q_red.prediction is True:
            conf_red = q_red.confidence
        else:
            # If prediction is False, confidence in "Glow" is 0.
            conf_red = 0.0
            
        history_red.append(conf_red)
        
        # Dashboard Log
        if i % 5 == 0:
            print(f"  Cycle {i}: Stable_Conf={conf_blue:.2f}, Volatile_Conf={conf_red:.2f}")

    return history_blue, history_red, cycles, kill_switch_cycle

def plot_results(history_blue, history_red, total_cycles, kill_event):
    print("[VISUALIZATION] Rendering Observability Graph...")
    cycles_x = np.arange(total_cycles)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Metrics
    ax.plot(cycles_x, history_blue, color='#00FFFF', linewidth=3, label='Stable Knowledge (Blue Rule)')
    ax.plot(cycles_x, history_red, color='#FF3333', linewidth=3, linestyle='--', label='Volatile Knowledge (Red Rule)')
    
    # Annotations
    ax.annotate('CONTEXT SHIFT\n(Kill Switch)', 
                xy=(kill_event, history_red[kill_event-1] if kill_event > 0 else 0), 
                xytext=(kill_event + 5, 0.8),
                arrowprops=dict(facecolor='white', shrink=0.05),
                fontsize=12, color='white', fontweight='bold')

    ax.set_title('CORTEX-Ω: Real-Time Epistemic Plasticity', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Axiom Confidence (0.0 - 1.0)', fontsize=12)
    ax.set_xlabel('Inference Cycles', fontsize=12)
    ax.set_ylim(-0.1, 1.2)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    filename = 'realtime_metrics.png'
    plt.savefig(filename, dpi=300)
    print(f"[SUCCESS] Graph saved to: {filename}")

if __name__ == "__main__":
    try:
        h_blue, h_red, cycles, k_switch = run_live_experiment()
        plot_results(h_blue, h_red, cycles, k_switch)
    except ImportError:
        print("[ERROR] 'cortex' library not found. Did you run 'pip install -e .' ?")
    except Exception as e:
        print(f"[ERROR] Runtime failure: {e}")
        import traceback
        traceback.print_exc()
