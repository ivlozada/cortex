import matplotlib.pyplot as plt
import numpy as np
from cortex import Cortex
import sys
import os

def run_strobe_test():
    print("\n=== CORTEX-Ω: The Strobe Light Protocol (High-Frequency Adaptability) ===\n")
    brain = Cortex(sensitivity=0.1)
    
    cycles = 100
    period = 20 # Cambiar la verdad cada 20 ciclos (10 ON, 10 OFF)
    
    history_confidence = []
    ground_truth_signal = []

    print(f"[TEST] Starting {cycles} cycles of rapid concept drift...")
    
    for i in range(cycles):
        # 1. Definir la Verdad del Momento (Onda Cuadrada)
        # Si estamos en la primera mitad del periodo, Rojo es VERDADERO. Si no, FALSO.
        current_truth = (i % period) < (period / 2)
        
        ground_truth_signal.append(1.0 if current_truth else 0.0)
        
        # 2. Alimentar al Cerebro
        # Nota: Usamos mass='heavy' como contexto constante
        brain.absorb_memory(
            [{'color': 'red', 'mass': 'heavy', 'result': current_truth}], 
            target_label='glow'
        )
        
        # 3. Consultar Confianza
        # Queremos saber cuánto confía en que "Rojo -> Glow"
        q = brain.query(color='red', mass='heavy')
        
        if q.prediction is True:
            conf = q.confidence
        else:
            # Si predice False, la confianza en "Glow" es 0
            conf = 0.0
            
        history_confidence.append(conf)
        
        # Log visual cada cambio de fase
        if i % (period/2) == 0:
            state = "ON " if current_truth else "OFF"
            print(f"   Cycle {i:03d}: Reality is {state} | Cortex Confidence: {conf:.2f}")

    return history_confidence, ground_truth_signal

def plot_strobe(history, signal):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(history))
    
    # La señal de la Realidad (Lo que debería ser)
    ax.step(x, signal, color='#333333', linewidth=4, label='Ground Truth (Reality)', where='post')
    
    # La respuesta de CORTEX (Lo que es)
    # Usamos step para resaltar la naturaleza digital
    ax.plot(x, history, color='#00FF00', linewidth=2, label='Cortex Adaptability')
    
    # Rellenar el área para que se vea "Cyberpunk"
    ax.fill_between(x, history, color='#00FF00', alpha=0.2)

    ax.set_title('The Strobe Light Protocol: High-Frequency Re-Learning', fontsize=16, fontweight='bold')
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_xlabel('Cycles (Time)', fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='upper right')
    
    filename = 'cortex_strobe_proof.png'
    plt.savefig(filename, dpi=300)
    print(f"\n[SUCCESS] Strobe graph generated: {filename}")
    print("   If the Green line matches the Grey line closely, you have beaten Deep Learning latency.")

if __name__ == "__main__":
    h, s = run_strobe_test()
    plot_strobe(h, s)
