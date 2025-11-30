from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .hypothesis import HypothesisGenerator
    # from .compiler import KnowledgeCompiler # If needed

@dataclass
class KernelConfig:
    """
    Configuración y dependencias del kernel.
    Es 'estado externo' que controla cómo propone parches y cómo evalúa armonía.
    """
    lambda_complexity: float = 0.3
    max_memory: int = 100
    # Simulated Annealing parameters
    temperature: float = 1.0
    cooling_rate: float = 0.95
    # Motor de generación de parches estructurales
    patch_generator: Optional['HypothesisGenerator'] = None
    # CORTEX-OMEGA: Compiler
    compiler: Optional[Any] = None # Avoid circular import for now
    # Refinement Loop
    max_refinement_steps: int = 3
    
    # CORTEX-OMEGA v1.5: World-Class Config
    mode: str = "robust" # "robust" (default) or "strict"
    priors: Dict[str, float] = field(default_factory=lambda: {"rule_base": 0.5, "exception": 0.3})
    noise_model: Dict[str, float] = field(default_factory=lambda: {"false_positive": 0.05, "false_negative": 0.05})
    plasticity: Dict[str, Any] = field(default_factory=lambda: {"min_conf_to_keep": 0.6, "max_rule_count": 500})
    feature_priors: Dict[str, float] = field(default_factory=dict)
    
    # CORTEX-OMEGA v1.6: Anti-Hallucination Constants
    complexity_alpha: float = 0.2
    complexity_base_cost: float = 0.5
    complexity_concept_cost: float = 0.8
    
    sa_min_temp: float = 1e-5
    sa_acceptance_threshold: float = 0.01
    
    inference_max_iterations: int = 1000
