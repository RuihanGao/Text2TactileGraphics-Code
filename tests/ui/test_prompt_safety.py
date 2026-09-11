"""Safety ordering, fail-closed behavior, and private submission logging."""

import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from text2tactilegraphics.ui.input_log import record_input
from text2tactilegraphics.ui.prompt_planner import (
    SAFETY_INSTRUCTION,
    GeminiPromptPlanner,
    InvalidPromptError,
    PlannerError,
    PromptPlan,
)
from text2tactilegraphics.ui.public_app import create_demo
from text2tactilegraphics.ui.quick_pipeline import QuickSettings, QuickState


@pytest.mark.parametrize(
    "response",
    ['{"safe":false}', "{}", '{"safe":"true"}', "", '{"safe":true,"extra":1}'],
)
def test_rejected_or_malformed_decision_never_parses(response, monkeypatch):
    from google import genai

    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-placeholder")
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.models.generate_content.return_value.text = response
    monkeypatch.setattr(genai, "Client", Mock(return_value=client))
    with pytest.raises((InvalidPromptError, PlannerError)):
        GeminiPromptPlanner().plan("untrusted prompt")
    assert client.models.generate_content.call_count == 1
    assert (
        client.models.generate_content.call_args.kwargs["config"].system_instruction
        == SAFETY_INSTRUCTION
    )


def test_safe_check_precedes_plan():
    planner = GeminiPromptPlanner()
    plan = PromptPlan(shape_prompt="lamp", regions=[], braille_label="lamp")
    planner._request = Mock(side_effect=[Mock(safe=True), plan])
    assert planner.plan("lamp") == plan
    assert planner._request.call_args_list[0].args[1] == SAFETY_INSTRUCTION
    assert planner._request.call_count == 2


def test_private_log_concurrent_and_multiline(tmp_path, monkeypatch):
    path = tmp_path / "private" / "inputs.jsonl"
    monkeypatch.setenv("TEXT2TACTILEGRAPHICS_INPUT_LOG", str(path))
    text = 'lamp\n{"forged":true} café'
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: record_input("generate", {"prompt": text}), range(40)))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 40
    assert len({r["submission_id"] for r in rows}) == 40
    assert all(r["inputs"]["prompt"] == text for r in rows)
    assert path.stat().st_mode & 0o777 == 0o600


def test_log_failure_stops_submission(tmp_path, monkeypatch):
    monkeypatch.setenv("TEXT2TACTILEGRAPHICS_INPUT_LOG", str(tmp_path))
    with pytest.raises(ValueError, match="Could not save your submission"):
        record_input("generate", {"prompt": "lamp"})


def test_ui_invalid_message_and_edit_does_not_mutate():
    planner = Mock()
    planner.plan.side_effect = InvalidPromptError()
    moderator = Mock()
    moderator.validate_prompt.side_effect = InvalidPromptError()
    demo = create_demo(planner=planner, moderator=moderator)
    callbacks = {f.fn.__name__: f.fn for f in demo.fns.values() if f.fn}
    result = list(callbacks["start"]("bad prompt"))[-1]
    assert (
        '<div role="status" aria-live="polite">Invalid prompt, please try again</div>'
        in result.values()
    )
    state = QuickState()
    old_plan = PromptPlan(shape_prompt="lamp", regions=[], braille_label="lamp")
    state.update_plan(old_plan)
    list(
        callbacks["edit"](
            "",
            None,
            state,
            None,
            "changed",
            [],
            "lamp",
            "apply",
            None,
            *QuickSettings().model_dump().values(),
        )
    )
    assert state.plan == old_plan
    assert state.status == "Invalid prompt, please try again"
    moderator.validate_prompt.assert_called_once()


def test_ui_records_before_admission(tmp_path, monkeypatch):
    path = tmp_path / "inputs.jsonl"
    monkeypatch.setenv("TEXT2TACTILEGRAPHICS_INPUT_LOG", str(path))
    demo = create_demo(planner=Mock())
    records = [
        f for f in demo.fns.values() if f.fn and f.fn.__name__ == "record_submission"
    ]
    records[0].fn("original\ninput", "")
    assert json.loads(path.read_text())["inputs"] == {"prompt": "original\ninput"}
    assert all(not f.queue for f in records)
