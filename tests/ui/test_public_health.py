from types import SimpleNamespace

from fastapi.testclient import TestClient

from text2tactilegraphics.ui.public_app import create_server


def test_health_does_not_load_models_and_reports_readiness(monkeypatch):
    from text2tactilegraphics.generation.models import ModelManager

    def fail(*a, **kw):
        raise AssertionError("Health must never load a model")

    monkeypatch.setattr(ModelManager, "_load_qwen_pipeline", fail)
    service = SimpleNamespace(state="Warming")
    app = create_server(service)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        service.state = "Ready"
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["ready"] is True
        assert set(response.json()) == {
            "process_alive",
            "gpu_count",
            "lifecycle",
            "state",
            "ready",
            "queue",
        }
        service.state = "Unhealthy"
        assert client.get("/readyz").status_code == 503
