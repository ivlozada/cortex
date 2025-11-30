from cortex_omega import Cortex

def test_david_vs_goliath():
    import random
    random.seed(42)
    
    brain = Cortex()

    # general rule: heavy things sink
    brain.absorb_memory([
        {"id": "obj_iron",  "material": "iron",  "is_heavy": True, "is_sink": True},
        {"id": "obj_lead",  "material": "lead",  "is_heavy": True, "is_sink": True},
        {"id": "obj_stone", "material": "stone", "is_heavy": True, "is_sink": True},
        {"id": "obj_plastic", "material": "plastic","is_heavy": False, "is_sink": False},
    ], target_label="sink")

    # exception: balsa is heavy but does NOT sink
    brain.absorb_memory([
        {"id": "obj_balsa", "material": "balsa", "is_heavy": True, "is_sink": False},
    ], target_label="sink")

    iron  = brain.query(material="iron",  is_heavy=True, target="sink")
    balsa = brain.query(material="balsa", is_heavy=True, target="sink")

    assert iron.prediction   is True,  f"Iron should sink, got {iron.prediction}. Reason: {iron.explanation}"
    assert balsa.prediction  is False, f"Balsa should not sink, got {balsa.prediction}. Reason: {balsa.explanation}"
