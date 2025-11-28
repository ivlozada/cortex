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
        return all(not a[0].isupper() for a in self.args)
    
    def __repr__(self):
        neg = "¬" if self.negated else ""
        return f"{neg}{self.predicate}({', '.join(self.args)})"


@dataclass
class Rule:
    """Una regla lógica: head :- body1, body2, ..."""
    id: str
    head: Literal
    body: List[Literal]
    confidence: float = 1.0
    support_count: int = 0
    failure_count: int = 0
    # CORTEX-OMEGA Pillar 4: Concept Compression
    usage_count: int = 0
    last_used: float = field(default_factory=lambda: 0.0)
    
    def __repr__(self):
        body_str = ", ".join(str(b) for b in self.body)
        conf_str = f" [{self.confidence:.2f}]" if self.confidence < 1.0 else ""
        return f"{self.id}: {self.head} :- {body_str}{conf_str}"
    
    def clone(self, new_id: str = None) -> 'Rule':
        """Copia profunda de la regla."""
        return Rule(
            id=new_id or f"{self.id}_clone",
            head=copy.deepcopy(self.head),
            body=copy.deepcopy(self.body),
            confidence=self.confidence,
            support_count=self.support_count,
            failure_count=self.failure_count,
            usage_count=self.usage_count,
            last_used=self.last_used
        )


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
        self.rules: Dict[str, Rule] = {}
        self.rules_by_head: Dict[str, List[str]] = defaultdict(list)
        self.version = 0
    
    def add(self, rule: Rule):
        """Añade una regla."""
        if rule.id in self.rules:
            self.remove(rule.id)
        self.rules[rule.id] = rule
        self.rules_by_head[rule.head.predicate].append(rule.id)
        self.version += 1
    
    def remove(self, rule_id: str):
        """Elimina una regla."""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            self.rules_by_head[rule.head.predicate].remove(rule_id)
            del self.rules[rule_id]
            self.version += 1
    
    def replace(self, old_id: str, new_rule: Rule):
        """Reemplaza una regla."""
        self.remove(old_id)
        self.add(new_rule)
    
    def get_rules_for_predicate(self, predicate: str) -> List[Rule]:
        """Obtiene todas las reglas que derivan un predicado."""
        return [self.rules[rid] for rid in self.rules_by_head.get(predicate, [])]
    
    def prune(self, threshold: float = 0.1) -> List[str]:
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
    
    paren_idx = s.index("(")
    predicate = s[:paren_idx]
    args_str = s[paren_idx+1:-1]
    args = tuple(a.strip() for a in args_str.split(","))
    
    return Literal(predicate, args, negated)


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
    
    return Rule(
        id=rule_id or f"R{hash(s) % 10000}",
        head=head,
        body=body
    )
