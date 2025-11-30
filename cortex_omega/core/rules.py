"""
Cortex Core: Rules & Facts
==========================
Data structures for symbolic knowledge representation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import copy
import time
import hashlib
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RuleID:
    uid: str
    
    def __str__(self):
        return self.uid
        
    def __repr__(self):
        return f"RuleID({self.uid})"
        
    def __eq__(self, other):
        if isinstance(other, str):
            return self.uid == other
        return isinstance(other, RuleID) and self.uid == other.uid
        
    def __hash__(self):
        return hash(self.uid)


@dataclass
class Literal:
    """Un átomo lógico: predicate(arg1, arg2, ...)"""
    predicate: str
    args: Tuple[str, ...]
    negated: bool = False
    
    def __hash__(self):
        return hash((self.predicate, self.args, self.negated))
    
    def __eq__(self, other):
        return (self.predicate == other.predicate and 
                self.args == other.args and 
                self.negated == other.negated)
    
    def ground(self, bindings: Dict[str, str]) -> 'Literal':
        """Sustituye variables por valores concretos."""
        new_args = tuple(bindings.get(a, a) for a in self.args)
        return Literal(self.predicate, new_args, self.negated)
    
    def is_ground(self) -> bool:
        """True si no tiene variables (args que empiezan con mayúscula)."""
        return all(not (isinstance(a, str) and a and a[0].isupper()) for a in self.args)
    
    def __repr__(self):
        neg = "¬" if self.negated else ""
        return f"{neg}{self.predicate}({', '.join(str(a) for a in self.args)})"


@dataclass
class Rule:
    """Una regla lógica: head :- body1, body2, ..."""
    id: RuleID
    head: Literal
    body: List[Literal]
    confidence: float = 1.0
    support_count: int = 0
    failure_count: int = 0
    # CORTEX-OMEGA v1.4: First-Class Rule Statistics
    fires_pos: int = 0
    fires_neg: int = 0
    
    # CORTEX-OMEGA Pillar 4: Concept Compression
    # CORTEX-OMEGA Pillar 4: Concept Compression
    usage_count: int = 0
    last_used: float = field(default_factory=lambda: 0.0)
    
    # CORTEX-OMEGA v1.5: World-Class Inspection
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    @property
    def coverage(self) -> int:
        return self.fires_pos + self.fires_neg
        
    @property
    def reliability(self) -> float:
        return self.fires_pos / (self.coverage + 1e-6)
        
    @property
    def complexity(self) -> int:
        """
        CORTEX-OMEGA v1.4: MDL Complexity.
        Length of the rule (body + head).
        """
        return len(self.body) + 1
    
    @property
    def is_safe(self) -> bool:
        """
        Returns True if all variables in the head appear in the body.
        """
        head_vars = set(a for a in self.head.args if isinstance(a, str) and a and a[0].isupper())
        body_vars = set()
        for lit in self.body:
            for a in lit.args:
                if isinstance(a, str) and a and a[0].isupper():
                    body_vars.add(a)
        return head_vars.issubset(body_vars)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        return self.id == other.id

    def __repr__(self):
        body_str = ", ".join(str(b) for b in self.body)
        conf_str = f" [{self.confidence:.2f}]" if self.confidence < 1.0 else ""
        return f"{self.id}: {self.head} :- {body_str}{conf_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the rule to a dictionary."""
        return {
            "id": str(self.id),
            "head": str(self.head),
            "body": [str(b) for b in self.body],
            "confidence": self.confidence,
            "stats": {
                "support": self.support_count,
                "failures": self.failure_count,
                "fires_pos": self.fires_pos,
                "fires_neg": self.fires_neg,
                "reliability": self.reliability,
                "coverage": self.coverage
            },
            "meta": {
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "usage_count": self.usage_count
            }
        }

    def clone(self, new_id: RuleID = None) -> 'Rule':
        """Copia profunda de la regla."""
        return Rule(
            id=new_id or RuleID(f"{self.id.uid}_clone"),
            head=copy.deepcopy(self.head),
            body=copy.deepcopy(self.body),
            confidence=self.confidence,
            support_count=self.support_count,
            failure_count=self.failure_count,
            fires_pos=self.fires_pos,
            fires_neg=self.fires_neg,
            usage_count=self.usage_count,
            last_used=self.last_used,
            created_at=self.created_at, # Inherit creation time if cloning? Or new? 
                                        # If it's a mutation, it's a new rule.
                                        # If it's a copy, it's the same.
                                        # Let's assume mutation for now, so maybe new time?
                                        # Actually, for now let's copy to preserve history if needed, 
                                        # but usually we want new timestamps for new hypotheses.
                                        # Let's just copy for now.
            updated_at=time.time()      # Always update updated_at
        )

    def is_subsumed_by(self, other: 'Rule') -> bool:
        """
        Returns True if 'self' is more specific than (or equal to) 'other'.
        (i.e., 'other' is more general).
        
        Logic:
        1. Heads must match (predicate and args).
        2. 'other' body must be a subset of 'self' body.
        """
        if self.head.predicate != other.head.predicate:
            return False
        # Args matching is tricky with variables, but for now assume exact match or normalized vars
        # If we assume canonical variable naming (X, Y, Z...), exact match works.
        if self.head.args != other.head.args:
            return False
            
        # Check if other.body is subset of self.body
        # (Every literal in other.body must be in self.body)
        other_lits = set(other.body)
        self_lits = set(self.body)
        
        return other_lits.issubset(self_lits)


class FactBase:
    """Base de hechos: almacena hechos ground con probabilidades opcionales."""
    
    def __init__(self):
        self.facts: Dict[str, Set[Tuple[str, ...]]] = defaultdict(set)
        self.probabilities: Dict[Tuple[str, Tuple], float] = {}
    
    def add(self, predicate: str, args: Tuple[str, ...], prob: float = 1.0):
        """Añade un hecho."""
        self.facts[predicate].add(args)
        if prob < 1.0:
            self.probabilities[(predicate, args)] = prob
    
    def remove(self, predicate: str, args: Tuple[str, ...]):
        """Elimina un hecho."""
        self.facts[predicate].discard(args)
        self.probabilities.pop((predicate, args), None)
    
    def query(self, predicate: str, args: Tuple[str, ...] = None) -> List[Tuple[str, ...]]:
        """Busca hechos que coincidan. None en args es wildcard."""
        if predicate not in self.facts:
            return []
        
        if args is None:
            return list(self.facts[predicate])
        
        results = []
        for fact_args in self.facts[predicate]:
            if len(fact_args) != len(args):
                continue
            match = True
            for fa, qa in zip(fact_args, args):
                if qa is not None and fa != qa:
                    match = False
                    break
            if match:
                results.append(fact_args)
        return results
    
    def contains(self, literal: Literal) -> bool:
        """Verifica si un literal ground está en la base."""
        exists = literal.args in self.facts.get(literal.predicate, set())
        # logger.debug(f"DEBUG: FactBase.contains({literal}) -> {exists} in {self.facts.get(literal.predicate, set())}")
        if literal.negated:
            return not exists
        return exists
    
    def get_all_predicates(self) -> Set[str]:
        """Retorna todos los predicados conocidos."""
        return set(self.facts.keys())
    
    def get_entities(self) -> Set[str]:
        """Retorna todas las entidades (constantes) en la base."""
        entities = set()
        for args_set in self.facts.values():
            for args in args_set:
                entities.update(args)
        return entities
    
    def to_dict(self) -> Dict[str, List[Tuple]]:
        """Serializa la base de hechos."""
        return {k: list(v) for k, v in self.facts.items()}
    
    def __repr__(self):
        lines = []
        for pred, args_set in self.facts.items():
            for args in args_set:
                lines.append(f"{pred}({', '.join(args)})")
        return "\n".join(lines)


class RuleBase:
    """Base de reglas: almacena y gestiona reglas lógicas."""
    
    def __init__(self):
        self.rules: Dict[RuleID, Rule] = {}
        self.rules_by_head: Dict[str, List[RuleID]] = defaultdict(list)
        self.version = 0
    
    def add(self, rule: Rule):
        """Añade una regla."""
        if rule.id in self.rules:
            self.remove(rule.id)
        self.rules[rule.id] = rule
        self.rules_by_head[rule.head.predicate].append(rule.id)
        self.version += 1
        # logger.debug(f"DEBUG: RuleBase added {rule.id}")
    
    def remove(self, rule_id: RuleID):
        """Elimina una regla."""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            self.rules_by_head[rule.head.predicate].remove(rule_id)
            del self.rules[rule_id]
            self.version += 1
    
    def replace(self, old_id: RuleID, new_rule: Rule):
        """Reemplaza una regla."""
        self.remove(old_id)
        self.add(new_rule)
    
    def get_rules_for_predicate(self, predicate: str) -> List[Rule]:
        """Obtiene todas las reglas que derivan un predicado."""
        return [self.rules[rid] for rid in self.rules_by_head.get(predicate, [])]
    
    def prune(self, threshold: float = 0.1) -> List[RuleID]:
        """
        CORTEX-OMEGA: Elimina reglas con baja utilidad.
        Utility = usage_count * confidence.
        Retorna IDs de reglas eliminadas.
        """
        to_remove = []
        for rule_id, rule in self.rules.items():
            utility = rule.usage_count * rule.confidence
            if utility < threshold and rule.usage_count > 0:
                to_remove.append(rule_id)
        
        for rid in to_remove:
            self.remove(rid)
            
        return to_remove

    def __repr__(self):
        return "\n".join(str(r) for r in self.rules.values())


@dataclass
class Scene:
    """Una escena con objetos y sus propiedades."""
    id: str
    facts: FactBase
    target_entity: str
    target_predicate: str
    ground_truth: bool  # ¿El target es verdadero según la regla oculta?
    target_args: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        if self.target_args is None:
            self.target_args = (self.target_entity,)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "facts": self.facts.to_dict(),
            "target_entity": self.target_entity,
            "target_predicate": self.target_predicate,
            "ground_truth": self.ground_truth,
            "target_args": self.target_args
        }



# === Funciones de Conveniencia ===

def parse_literal(s: str) -> Literal:
    """Parsea una cadena como 'predicate(arg1, arg2)' o '¬predicate(arg1)'."""
    s = s.strip()
    negated = s.startswith("¬") or s.startswith("not ")
    if negated:
        s = s.lstrip("¬").lstrip("not ").strip()
    
    # Check for infix operators (CORTEX-OMEGA addition)
    operators = [">=", "<=", "!=", "=", ">", "<"]
    for op in operators:
        if op in s and "(" not in s:
            parts = s.split(op)
            if len(parts) == 2:
                return Literal(op, (parts[0].strip(), parts[1].strip()), negated)

    try:
        paren_idx = s.index("(")
        predicate = s[:paren_idx]
        args_str = s[paren_idx+1:-1]
        args = tuple(a.strip() for a in args_str.split(","))
        return Literal(predicate, args, negated)
    except ValueError:
        # Fallback for 0-arity atoms
        return Literal(s, (), negated)


def make_rule_id(rule: Rule) -> RuleID:
    """Generates a deterministic ID based on rule content."""
    head_str = str(rule.head)
    body_strs = sorted([str(b) for b in rule.body])
    payload = f"{head_str} :- {', '.join(body_strs)}"
    h = hashlib.sha1(payload.encode()).hexdigest()[:12]
    return RuleID(uid=f"R_{h}")

def parse_rule(s: str, rule_id: str = None) -> Rule:
    """Parsea una cadena como 'head :- body1, body2'."""
    parts = s.split(":-")
    head = parse_literal(parts[0].strip())
    
    body = []
    if len(parts) > 1 and parts[1].strip():
        # Parsing simple: split por coma, respetando paréntesis
        body_str = parts[1].strip()
        body_parts = []
        depth = 0
        current = ""
        for c in body_str:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif c == "," and depth == 0:
                body_parts.append(current.strip())
                current = ""
                continue
            current += c
        if current.strip():
            body_parts.append(current.strip())
        
        body = [parse_literal(bp) for bp in body_parts]
    
    # Create temporary rule to generate ID if not provided
    temp_rule = Rule(id=RuleID("temp"), head=head, body=body)
    rid = RuleID(rule_id) if rule_id else make_rule_id(temp_rule)
    
    return Rule(
        id=rid,
        head=head,
        body=body
    )
