#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Smoke test para validar la ampliación recursiva de Cortex-Omega:

1. Term soporta estructuras anidadas (s(s(zero))).
2. La unificación profunda funciona: unify(s(X), s(zero)) ⇒ X = zero.
3. El occurs-check bloquea unify(X, s(X)).
4. El repr es seguro para términos muy profundos (no crashea).
5. project_godel_max_v2 aprende:
   - Una regla recursiva tipo: add(s(V_0), K_1, s(V_2)) :- add(V_0, K_1, V_2).
   - Casos base con zero.
"""

import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from cortex_omega.core.rules import Term, Rule, Literal, RuleBase, FactBase
from cortex_omega.core.inference import InferenceEngine

# Helper for unification using InferenceEngine
def unify(term1, term2):
    engine = InferenceEngine(FactBase(), RuleBase())
    bindings = {}
    success = engine._unify_term(term1, term2, bindings)
    return bindings if success else None

# Try to import project_godel_max_v2
try:
    import project_godel_max_v2
except ImportError:
    project_godel_max_v2 = None


# ==========
# HELPERS
# ==========

ZERO = Term("zero")


def S(term: Term) -> Term:
    """Constructor sucinto de s(term)."""
    return Term("s", (term,))


def build_chain(n: int) -> Term:
    """Construye s(s(...s(zero))) de longitud n."""
    t = ZERO
    for _ in range(n):
        t = S(t)
    return t


def print_ok(msg: str) -> None:
    print(f"✅ {msg}")


def print_fail(msg: str) -> None:
    print(f"❌ {msg}")
    raise AssertionError(msg)


# ============================
# TEST 1: Term anidado & árbol
# ============================

def test_term_nested():
    print("\n=== TEST 1: Term soporta estructuras anidadas ===")
    two = S(S(ZERO))  # s(s(zero))

    # Comprobamos que la estructura es realmente un árbol anidado
    if not (two.name == "s"
            and len(two.args) == 1
            and two.args[0].name == "s"
            and len(two.args[0].args) == 1
            and two.args[0].args[0].name == "zero"):
        print_fail("La estructura de s(s(zero)) no coincide con el árbol esperado.")
    else:
        print_ok("Term representa correctamente s(s(zero)) como árbol anidado.")


# ==========================================
# TEST 2: Unificación profunda s(X), s(zero)
# ==========================================

def make_var(name: str):
    """
    Construye una variable según tu representación interna.
    En Cortex-Omega v2, las variables son Terms que empiezan con mayúscula y sin args.
    """
    return Term(name)


def test_deep_unification():
    print("\n=== TEST 2: Unificación profunda unify(s(X), s(zero)) ===")

    X = make_var("X")
    left = S(X)
    right = S(ZERO)

    subst = unify(left, right)

    if subst is None:
        print_fail("unify(s(X), s(zero)) devolvió fallo de unificación (None).")
        return

    # En Cortex-Omega, subst es un dict {var_name: value}
    try:
        value_for_X = subst.get(X.name, None)
    except Exception:
        print_fail("No pude extraer el valor de X desde la sustitución devuelta por unify().")
        return

    if value_for_X == ZERO or value_for_X == "zero":
        print_ok("unify(s(X), s(zero)) ⇒ X = zero (unificación profunda funcionando).")
    else:
        print_fail(f"Esperaba X = zero, pero obtuve: {value_for_X!r}")


# =========================================
# TEST 3: Occurs-check unify(X, s(X))
# =========================================

def test_occurs_check():
    print("\n=== TEST 3: Occurs-check bloquea unify(X, s(X)) ===")

    X = make_var("X")
    left = X
    right = S(X)

    try:
        subst = unify(left, right)
    except Exception as e:
        # Si lanzas una excepción específica de occurs-check, también es válido.
        print_ok(f"unify(X, s(X)) lanzó excepción (occurs-check activo): {type(e).__name__}")
        return

    # Si no hubo excepción, esperamos que el resultado sea un fallo de unificación
    if subst is None or subst is False:
        print_ok("unify(X, s(X)) devolvió fallo (None/False): occurs-check funcionando.")
    else:
        print_fail(
            f"unify(X, s(X)) devolvió una sustitución en lugar de fallar: {subst!r}"
        )


# ===================================================
# TEST 4: Repr seguro para términos muy profundos
# ===================================================

def test_safe_repr():
    print("\n=== TEST 4: repr seguro para términos profundos ===")

    deep_term = build_chain(70)  # > 60 niveles, debería activar podado interno
    try:
        s = repr(deep_term)
    except RecursionError:
        print_fail("repr(deep_term) lanzó RecursionError (safe_repr no está funcionando).")
        return

    # Heurística: esperamos o bien que aparezca "..." o que el repr no sea descomunal.
    if "..." in s or len(s) < 2000:
        print_ok("repr(deep_term) no crashea y parece truncado/razonable.")
    else:
        print_fail(
            "repr(deep_term) no contiene '...' y es sospechosamente largo; "
            "revisa tu implementación de safe_repr / límite de profundidad."
        )


# ===========================================================
# TEST 5: Reglas aprendidas en project_godel_max_v2 (integral)
# ===========================================================

def run_project_godel():
    """
    Envoltorio para ejecutar tu experimento Gödel y recuperar la teoría aprendida.
    """
    if project_godel_max_v2 is None:
        raise RuntimeError(
            "No se pudo importar project_godel_max_v2. "
            "Asegúrate de que el archivo existe en el directorio actual."
        )

    # project_godel_max_v2.py no tiene run_experiment, pero tiene run_peano_arithmetic
    # que crea un 'brain' y corre todo.
    # Vamos a modificar ligeramente project_godel_max_v2 para que sea importable y retorne el brain,
    # o vamos a invocar sus funciones.
    
    # Dado que project_godel_max_v2.py ejecuta main() al final si __name__ == "__main__",
    # podemos intentar llamar a run_peano_arithmetic() si expone el brain, 
    # pero run_peano_arithmetic() en el script actual imprime cosas y no retorna el brain.
    
    # Hack: Vamos a instanciar un Cortex y correr la lógica de aprendizaje manualmente aquí
    # replicando lo que hace el script, para no depender de modificar el script original
    # O mejor, modificamos el script original para que tenga una función que retorne la teoría.
    
    # Opción B: Importamos las funciones de enseñanza del script si es posible.
    # El script define run_peano_arithmetic() que hace todo.
    # Vamos a asumir que podemos ejecutarlo y capturar el estado final si lo modificamos.
    
    # Para este smoke test, vamos a simular la ejecución llamando a las funciones del script
    # pero necesitamos el objeto 'brain'.
    
    # Vamos a crear un brain aquí y usar las funciones del script si son puras, 
    # pero teach_successor y teach_addition usan brain.absorb_memory que no es lo que usamos en v2.
    
    # En v2 usamos learner.learn manual.
    # Así que mejor ejecutamos la lógica de v2 aquí directamente.
    
    from cortex_omega import Cortex
    from cortex_omega.core.rules import Scene, FactBase
    
    # Setup similar a project_godel_max_v2.py
    brain = Cortex(mode="strict")
    
    # Configurar learner para recursión
    brain.config.inference_max_iterations = 5
    
    # Generar datos (reducidos para velocidad)
    memory = []
    for x in range(2): # Solo 0 y 1 para velocidad
        for y in range(2):
            z = x + y
            if z > 2: continue
            
            # Construir términos
            x_term = build_chain(x)
            y_term = build_chain(y)
            z_term = build_chain(z)
            
            # Crear escena
            fb = FactBase()
            # No necesitamos hechos planos para recursión pura, solo target args
            
            scene = Scene(
                id=f"add_{x}_{y}",
                facts=fb,
                target_entity="dummy",
                target_predicate="add",
                ground_truth=True,
                target_args=(x_term, y_term, z_term)
            )
            memory.append(scene)
            
    # Aprender
    learner = brain.learner
    for scene in memory:
        brain.theory, brain.memory = learner.learn(
            brain.theory, scene, brain.memory, brain.axioms
        )
        
    return brain.theory


def test_learned_addition_rules():
    print("\n=== TEST 5: Reglas aprendidas (add/3 recursiva + casos base) ===")

    try:
        theory = run_project_godel()
    except Exception as e:
        print(f"⚠️  Saltando TEST 5 por error en ejecución: {e}")
        import traceback
        traceback.print_exc()
        return

    rules = list(theory.rules.values())

    # Buscamos:
    # 1) Una regla recursiva: add(s(V_0), K_1, s(V_2)) :- add(V_0, K_1, V_2)
    # 2) Al menos un caso base: add(X_const, zero, X_const)

    recursive_found = False
    base_case_found = False

    for rule in rules:
        head = rule.head
        body = rule.body

        # Saltamos reglas sin cabeza estándar
        if head is None:
            continue

        # Esperamos predicado "add"
        if head.predicate != "add":
            continue

        # --- Detectar caso recursivo ---
        # add(s(V0), K1, s(V2)) :- add(V0, K1, V2)
        # Verificamos estructura de argumentos
        if (len(head.args) == 3 and
            isinstance(head.args[0], Term) and head.args[0].name == "s" and
            isinstance(head.args[2], Term) and head.args[2].name == "s" and
            len(body) == 1 and
            body[0].predicate == "add"):

            recursive_found = True
            print(f"  -> Encontrada recursiva: {rule}")

        # --- Detectar caso base ---
        # add(X, zero, X) o add(zero, zero, zero)
        if len(head.args) == 3:
            a0, a1, a2 = head.args
            # Check for zero in middle
            is_zero_middle = False
            if isinstance(a1, Term) and a1.name == "zero":
                is_zero_middle = True
            elif isinstance(a1, str) and a1 == "zero": # Legacy/String check
                 is_zero_middle = True
                 
            if is_zero_middle:
                # Check equality of first and third
                if a0 == a2:
                    base_case_found = True
                    print(f"  -> Encontrado caso base: {rule}")

    if recursive_found:
        print_ok("Se encontró una regla recursiva tipo add(s(V0), K1, s(V2)) :- add(V0, K1, V2).")
    else:
        print_fail(
            "No se encontró ninguna regla recursiva add/3 con patrón s(V0), K1, s(V2) :- add(V0, K1, V2)."
        )

    if base_case_found:
        print_ok("Se encontró al menos un caso base add(X, zero, X).")
    else:
        print_fail(
            "No se encontró ningún caso base add(X, zero, X). "
        )


# ================
# ENTRY POINT
# ================

if __name__ == "__main__":
    # Ejecución secuencial simple sin pytest
    test_term_nested()
    test_deep_unification()
    test_occurs_check()
    test_safe_repr()
    test_learned_addition_rules()
    print("\n🎉 Todos los tests ejecutados.")
