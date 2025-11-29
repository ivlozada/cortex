"""
Cortex-Omega Error Taxonomy
===========================
Defines the custom exception hierarchy for the library.
"""

class CortexError(Exception):
    """Base class for all Cortex-Omega exceptions."""
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
