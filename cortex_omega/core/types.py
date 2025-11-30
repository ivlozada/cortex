from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from .rules import Rule, FactBase, Scene

class PatchOperation(Enum):
    """Operaciones de modificación de reglas."""
    ADD_LITERAL = "ADD_LITERAL"           # Añadir condición positiva
    ADD_NEGATED_LITERAL = "ADD_NEG"       # Añadir condición negativa
    REMOVE_LITERAL = "REMOVE"             # Quitar condición
    REPLACE_LITERAL = "REPLACE"           # Cambiar condición
    ADD_EXCEPTION = "ADD_EXCEPTION"       # Añadir cláusula de excepción
    CREATE_BRANCH = "CREATE_BRANCH"       # Crear regla alternativa (disyunción)
    CREATE_CONCEPT = "CREATE_CONCEPT"     # Crear concepto intermedio reutilizable
    SEQUENCE = "SEQUENCE"                 # Secuencia de parches

@dataclass
class Patch:
    """Un parche propuesto para una regla."""
    operation: PatchOperation
    target_rule_id: str
    details: Dict[str, Any]
    confidence: float = 0.5
    explanation: str = ""
    source_strategy: str = ""
    
    def __repr__(self):
        return f"Patch({self.operation.value}, conf={self.confidence:.2f}): {self.explanation}"

@dataclass
class FailureContext:
    """Contexto completo de un fallo de predicción."""
    rule: Rule
    error_type: str  # FALSE_POSITIVE o FALSE_NEGATIVE
    target_entity: str
    scene_facts: FactBase
    prediction: bool
    ground_truth: bool
    target_predicate: str = "" # Added for Anti-Unification
    target_args: Optional[Tuple[str, ...]] = None # Added for Binary Relations
    inference_trace: List[Dict] = field(default_factory=list)
    memory: List[Scene] = field(default_factory=list)
    
    # CORTEX-OMEGA v1.5: Feature Priors
    feature_priors: Dict[str, float] = field(default_factory=dict)
