from cortex_omega import Cortex
import os
import tempfile

def test_save_load_brain_roundtrip(tmp_path: "os.PathLike[str]" = None):
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()

    brain = Cortex()
    data = [
        {"id": "iron", "mass": "heavy", "sinks": True},
        {"id": "wood", "mass": "light", "sinks": False},
    ]
    brain.absorb_memory(data, target_label="sinks")

    # Original query
    q1 = brain.query(id="iron", mass="heavy", target="sinks")
    assert q1.prediction is True

    # Save & load
    path = os.path.join(tmp_path, "brain.pkl")
    brain.save_brain(path)
    brain2 = Cortex.load_brain(path)

    q2 = brain2.query(id="iron", mass="heavy", target="sinks")
    assert q2.prediction is True
    # And the confidence should be very close
    assert abs(q2.confidence - q1.confidence) < 1e-6
