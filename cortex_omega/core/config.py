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
    
    # Debug Mode
    debug: bool = False
    
    # CI Stability Fix (v2.0.1)
    random_seed: Optional[int] = None

    # CORTEX-OMEGA v2.0: Centralized Hyperparameters
    hyperparams: 'Hyperparameters' = field(default_factory=lambda: Hyperparameters())
    
    def __post_init__(self):
        if self.feature_priors:
            self.hyperparams.feature_weights.update(self.feature_priors)

@dataclass
class Hyperparameters:
    """
    Centralized hyperparameters for the engine.
    Replaces magic numbers in hypothesis generation and inference.
    """
    # Hypothesis Generation
    temporal_confidence_threshold: float = 0.9
    relational_confidence_threshold: float = 0.55
    min_rule_score: float = 0.01
    
    # Feature Priorities (formerly PROPERTY_PRIORITY)
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "material": 10.0,
        "shape": 5.0,
        "color": 2.0,
        "size": 2.0,
        "texture": 2.0
    })
    
    # Inference & Learning
    strict_mode_penalty: int = 1_000_000
    default_failure_penalty: int = 1
    bayesian_alpha: float = 1.0
    bayesian_beta: float = 2.0
