"""
Cortex-Omega Error Taxonomy
===========================
Defines the custom exception hierarchy for the library.
"""

class CortexError(Exception):
    """Base class for all Cortex-Omega exceptions."""
    pass

class GDMError(CortexError):
    """Base class for GDM Kernel specific errors."""
    pass

class DataFormatError(CortexError):
    """Raised when input data format is invalid or unrecognizable."""
    pass

class RuleParseError(CortexError):
    """Raised when a rule string cannot be parsed."""
    pass

class InferenceTimeoutError(CortexError):
    """Raised when inference exceeds the allowed time budget."""
    pass

class ConfigurationError(CortexError):
    """Raised when the kernel configuration is invalid."""
    pass

class EpistemicVoidError(CortexError):
    """Raised when the engine has absolutely no knowledge to answer a query."""
    pass

class InconsistentTheoryError(GDMError):
    """Raised when the theory contains logical contradictions."""
    pass

class LearningConvergenceError(GDMError):
    """Raised when the learner fails to converge to a satisfactory theory."""
    pass
