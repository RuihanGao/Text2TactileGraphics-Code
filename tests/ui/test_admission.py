import json
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

from text2tactilegraphics.ui.admission import Admission


def test_bounded_duplicate_and_forged_tickets(tmp_path):
    gate = Admission(max_pending=2, telemetry=tmp_path / "events.jsonl")
    a = gate.reserve("a")
    with pytest.raises(ValueError, match="already"):
        gate.reserve("a")
    gate.reserve("b")
    gate.reserve("c")
    with pytest.raises(ValueError, match="busy"):
        gate.reserve("d")
    with pytest.raises(ValueError, match="no longer"):
        with gate.run(a, "attacker"):
            pytest.fail("Forged owner ran")
    with gate.run(a, "a"):
        assert gate.snapshot()["active"] == 1
    assert not gate.owns("a")
    assert "attacker" not in (tmp_path / "events.jsonl").read_text()


def test_wait_budget_and_failure_recovery():
    gate = Admission(max_pending=3, wait_limit=60, initial_seconds=50)
    a = gate.reserve("a")
    b = gate.reserve("b")
    with pytest.raises(ValueError, match="busy"):
        gate.reserve("c")
    gate.jobs[b]["accepted"] -= 61
    assert not gate.owns("b")
    with pytest.raises(RuntimeError):
        with gate.run(a, "a"):
            raise RuntimeError("generation failed")
    with gate.run(gate.reserve("a"), "a"):
        pass


def test_whole_request_serialization_and_cancellation():
    gate = Admission()
    tokens = [gate.reserve(str(i)) for i in range(3)]
    active, peak = 0, 0
    barrier = threading.Barrier(3)

    def run(i):
        nonlocal active, peak
        barrier.wait()
        with gate.run(tokens[i], str(i)):
            active += 1
            peak = max(peak, active)
            time.sleep(0.02)
            active -= 1

    with ThreadPoolExecutor(3) as pool:
        list(pool.map(run, range(3)))
    assert peak == 1
    assert not gate.jobs


def test_wrapped_error_class_and_visible_expiry(tmp_path):
    path = tmp_path / "events.jsonl"
    gate = Admission(telemetry=path)
    token = gate.reserve("a")
    with pytest.raises(RuntimeError):
        with gate.run(token, "a"):
            try:
                raise MemoryError("simulated allocation failure")
            except MemoryError as exc:
                raise RuntimeError("handler wrapper") from exc
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[-1]["error_class"] == "MemoryError"
    token = gate.reserve("b")
    gate.jobs[token]["accepted"] -= 181
    assert not gate.owns("b")
    assert "wait limit" in gate.notice("b")

    def cancelled():
        with gate.run(gate.reserve("cancel"), "cancel"):
            yield

    iterator = cancelled()
    next(iterator)
    iterator.close()
    assert not gate.jobs


def test_request_completion_collects_unreachable_cycles():
    class Geometry:
        pass

    gate = Admission()
    with gate.run(gate.reserve("visitor"), "visitor"):
        mesh = Geometry()
        mesh.self_reference = mesh
        reference = weakref.ref(mesh)
        del mesh
        assert reference() is not None
    assert reference() is None
