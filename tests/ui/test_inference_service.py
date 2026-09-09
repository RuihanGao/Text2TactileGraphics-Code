from types import SimpleNamespace

import pytest

from text2tactilegraphics.ui import inference_service as service_module


def test_readiness_only_after_successful_warmup(monkeypatch):
    service = service_module.InferenceService()
    with pytest.raises(ValueError, match="warming"):
        service.require_ready()
    events = []
    manager = SimpleNamespace(
        config=SimpleNamespace(model_lifecycle="dual_a100_80gb"),
        warmup=lambda: events.append("models"),
    )
    monkeypatch.setattr(service_module, "global_model_manager", lambda: manager)

    def generate(*args):
        assert service.state == "Warming"
        events.append("kernels")
        yield None

    monkeypatch.setattr(service_module.QuickPipeline, "generate", generate)
    service.warmup()
    service.require_ready()
    assert events == ["models", "kernels"]
    service.warmup()
    assert events == ["models", "kernels"]


def test_failed_warmup_does_not_accept_requests(monkeypatch):
    def fail():
        raise RuntimeError("private diagnostic")

    monkeypatch.setattr(service_module, "global_model_manager", fail)
    service = service_module.InferenceService()
    service.warmup()
    assert service.state == "Unhealthy"
    assert service.error_class == "RuntimeError"
    with pytest.raises(ValueError):
        service.require_ready()
