"""
Cortex IO: Universal Ingestor
=============================
Ingests data from various sources (CSV, JSON) and converts it to Cortex Scenes.
Includes basic heuristics for data cleaning and normalization.
"""

import csv
import os
from typing import List, Dict, Any
from ..core.rules import Scene, FactBase

class SmartIngestor:
    """
    Ingestor inteligente que detecta formato y normaliza datos.
    """
    
    def ingest(self, filepath: str, target_col: str = "IS_FRAUD") -> List[Scene]:
        """
        Ingesta un archivo y retorna una lista de escenas.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        if filepath.endswith(".csv"):
            return self._ingest_csv(filepath, target_col)
        else:
            raise ValueError("Unsupported file format. Only CSV supported for now.")
            
    def _ingest_csv(self, filepath: str, target_col: str) -> List[Scene]:
        scenes = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Generate a unique ID if not present
                tx_id = row.get("TransactionID", f"TX_{len(scenes):03d}")
                
                facts = FactBase()
                target_val = False
                
                for col, val in row.items():
                    if col == target_col:
                        # Parse target
                        target_val = val.lower() in ["yes", "true", "1"]
                        continue
                        
                    if col == "TransactionID":
                        continue
                        
                    # Heuristic Normalization
                    norm_val = self._normalize(col, val)
                    
                    # Add to facts
                    # Metaphor mapping (optional, but keeps the demo cool)
                    pred = col.lower()
                    if "amount" in pred:
                        pred = "mass"
                    elif "user" in pred:
                        pred = "type"
                    elif "location" in pred:
                        pred = "loc"
                        
                    facts.add(pred, (tx_id, norm_val))
                    
                # Create Scene
                # We assume target predicate is "fraud" if target_col is IS_FRAUD
                target_pred = "fraud" if "fraud" in target_col.lower() else target_col.lower()
                
                scene = Scene(tx_id, facts, tx_id, target_pred, target_val)
                scenes.append(scene)
                
        return scenes
    
    def _normalize(self, col: str, val: str) -> str:
        """
        Normaliza valores sucios.
        """
        val = val.strip()
        
        # Money Heuristic
        if "$" in val or "amount" in col.lower():
            clean_val = val.replace("$", "").replace(",", "")
            try:
                fval = float(clean_val)
                # Binning simple for demo
                return "heavy" if fval > 10000 else "light"
            except ValueError:
                pass
                
        # Text Normalization
        return val.lower()
