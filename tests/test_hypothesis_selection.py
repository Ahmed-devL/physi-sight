"""Training-only hypothesis selection and deterministic-pipeline ablations."""
import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import backend.agent as agent
import backend.solver as solver


def drag_data(*, mixed=True):
    t = np.arange(120) / 30
    terms = ("constant", "velocity", "quadratic_drag") if mixed else ("constant", "quadratic_drag")
    coefficients = (-9.81, -.8, -.18) if mixed else (-9.81, -.18)
    values = solver.integrate(t, terms, coefficients, (12, 1.5))
    return t, values + np.random.default_rng(16).normal(0, .0001, len(t))


def select(t, y, **kwargs):
    return asyncio.run(agent.select_hypotheses(t, y, regime="drag_fall", noise_scale=.0001,
                                              release_index=0, **kwargs))


@pytest.mark.parametrize("proposal", [None, [], {}, {"candidate": "mixed_drag"},
    {"candidate": "nonlinear_pendulum", "rationale": "wrong regime"},
    {"candidate": "mixed_drag", "rationale": "test", "coefficients": [-9.81]},
    {"candidate": ["mixed_drag"], "rationale": "list"},
    {"candidate": "exec('arbitrary code')", "rationale": "code"},
    {"candidate": "mixed_drag", "rationale": ""}])
def test_strict_proposal_validation(proposal):
    with pytest.raises(ValueError):
        agent.validate_proposal(proposal, "drag_fall")


@pytest.mark.parametrize("reply", ["not JSON", "[]", '{"candidate":"unknown","rationale":"x"}'])
def test_malformed_model_response_is_rejected(monkeypatch, reply):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]))
    monkeypatch.setattr(agent, "_hypothesis_client", lambda p: (client, "muse-wonder-1.0"))
    with pytest.raises(ValueError):
        asyncio.run(agent.hypothesis_loop({"regime": "drag_fall"}, "meta"))


@pytest.mark.parametrize("failure", [RuntimeError("API unavailable"), ValueError("Malformed JSON"),
                                   TimeoutError("network timeout")])
def test_wonder_failures_retain_original_winner(monkeypatch, failure):
    t, y = drag_data(mixed=False)
    monkeypatch.setattr(agent, "hypothesis_loop", AsyncMock(side_effect=failure))
    _, off, _ = select(t, y, wonder_enabled=False)
    _, failed, record = select(t, y)
    assert failed.name == off.name == "quadratic_drag"
    assert failed.coefficients == off.coefficients
    np.testing.assert_array_equal(failed.prediction, off.prediction)
    assert record["fallback_used"] is True
    assert record["proposal_fit"] is None
    assert record["wonder_proposal"]["status"] == "failed"


def test_invalid_proposal_fails_open_and_off_never_calls_api(monkeypatch):
    t, y = drag_data()
    propose = AsyncMock(return_value={"candidate": "free_fall", "rationale": "x", "code": "x"})
    monkeypatch.setattr(agent, "hypothesis_loop", propose)
    _, off, off_record = select(t, y, wonder_enabled=False)
    propose.assert_not_awaited()
    assert off_record["wonder_proposal"]["status"] == "disabled"
    _, failed, record = select(t, y)
    assert failed.name == off.name
    assert record["fallback_used"]


@pytest.mark.parametrize("mixed,expected_change", [(True, True), (False, False)])
def test_ablation_changes_hypothesis_set_and_reports_equation_separately(monkeypatch, mixed, expected_change):
    t, y = drag_data(mixed=mixed)
    monkeypatch.setattr(agent, "hypothesis_loop", AsyncMock(return_value={
        "candidate": "mixed_drag", "rationale": "Inspect training residuals"}))
    off_results, off, off_record = select(t, y, wonder_enabled=False)
    results, winner, record = select(t, y)
    assert "mixed_drag" not in [r.name for r in off_results]
    assert "mixed_drag" in [r.name for r in results]
    verdict = record["bic_verdict"]
    assert "mixed_drag" in verdict["evaluated_hypotheses"]
    assert verdict["winning_equation_changed"] is expected_change
    assert (winner.name != off.name) is expected_change
    assert record["baseline_fit"] == off_record["baseline_fit"]
    assert record["heldout_evaluation"]["discovery_improvement_claimed"] is False
    if mixed:
        assert winner.name == "mixed_drag"
        for term, true in zip(("constant", "velocity", "quadratic_drag"), (-9.81, -.8, -.18)):
            assert winner.coefficients[term] <= 0
            assert winner.coefficients[term] == pytest.approx(true, rel=.02)
        assert record["heldout_evaluation"]["selected"]["rmse"] < record["heldout_evaluation"]["baseline"]["rmse"]
    print(json.dumps({"hypothesis_set_changed": True, "winning_equation_changed": expected_change,
                      "baseline": off.name, "selected": winner.name, "verdict": verdict, "heldout": record["heldout_evaluation"]}))


def test_holdout_cannot_affect_prompt_proposal_fitting_or_selection(monkeypatch):
    t, y = drag_data()
    prompts = []
    evaluated = []
    original_evaluate = agent.evaluate_fit
    def evaluate(*args):
        evaluated.append(True)
        return original_evaluate(*args)
    async def propose(evidence, provider):
        assert not evaluated  # No held-out evaluation before proposing/selecting.
        assert "heldout" not in json.dumps(evidence)
        assert set(evidence) == {"regime", "baseline_fit", "candidate_library"}
        assert evidence["baseline_fit"]["name"] == "free_fall"
        prompts.append(evidence)
        return {"candidate": "mixed_drag", "rationale": "Training residuals"}
    monkeypatch.setattr(agent, "hypothesis_loop", propose)
    monkeypatch.setattr(agent, "evaluate_fit", evaluate)
    before, winner, record = select(t, y)
    evaluated.clear()
    changed = y.copy()
    changed[84:] += 1000
    after, winner_after, changed_record = select(t, changed)
    assert prompts[0] == prompts[1]
    assert [(r.name, r.bic, r.coefficients) for r in before] == [(r.name, r.bic, r.coefficients) for r in after]
    assert winner.name == winner_after.name
    assert record["bic_verdict"] == changed_record["bic_verdict"]
    assert record["heldout_evaluation"] != changed_record["heldout_evaluation"]


def test_sparse_rejects_wrong_sign_missing_gravity_and_confounded_terms():
    t, y = drag_data()
    for proposal in ({"constant": 9.81}, {"velocity": -.2},
                     {"constant": -9.81, "velocity": .2}, {"position": -1},
                     {"constant": float("nan")}):
        assert not solver.valid_sparse_structure("drag", proposal, t[:84], y[:84])
    assert not solver.valid_sparse_structure("pendulum", {"position": -1, "sin_position": -1}, t[:84], y[:84])
    assert not solver.valid_sparse_structure("drag", {"constant": -9.81, "velocity": -.2}, t, 2*t)
    assert solver.valid_sparse_structure("drag", {"constant": -9.81, "velocity": -.8,
                                                 "quadratic_drag": -.18}, t[:84], y[:84])


def test_training_mode_never_calculates_heldout_metrics(monkeypatch):
    t, y = drag_data()
    monkeypatch.setattr(solver, "evaluate_fit", lambda *a: pytest.fail("Premature heldout evaluation"))
    results = solver.fit_trajectory(t, y, regime="drag", release_index=0, evaluate_heldout=False)
    assert all(r.heldout_rmse is None and r.heldout_r2 is None for r in results)


def test_existing_library_proposal_does_not_expand_candidates(monkeypatch):
    t, y = drag_data()
    monkeypatch.setattr(agent, "hypothesis_loop", AsyncMock(return_value={
        "candidate": "free_fall", "rationale": "Check gravity only"}))
    off_results, off, _ = select(t, y, wonder_enabled=False)
    results, best, record = select(t, y)
    assert [r.name for r in results] == [r.name for r in off_results]
    assert best.name == off.name
    assert record["bic_verdict"]["proposal_already_in_deterministic_library"]
    assert record["sparse_validation"] == []
    assert record["heldout_evaluation"]["baseline"] == record["heldout_evaluation"]["selected"]



def test_wonder_off_is_forwarded_by_existing_routes(monkeypatch):
    from fastapi.testclient import TestClient
    import backend.app as api
    analyze = AsyncMock(return_value={"regime": "pendulum"})
    monkeypatch.setattr(api, "analyze_video", analyze)
    response = TestClient(api.app).post('/api/analyze/demo/damped_pendulum.mp4?wonder_enabled=false')
    assert response.status_code == 200
    assert analyze.call_args.kwargs["wonder_enabled"] is False


def test_proposal_numerical_failure_falls_back(monkeypatch):
    t, y = drag_data()
    real_fit = agent.fit_trajectory
    def fit(*args, **kwargs):
        if kwargs.get("candidate") == "mixed_drag":
            raise RuntimeError("Optional numerical candidate failed")
        return real_fit(*args, **kwargs)
    monkeypatch.setattr(agent, "fit_trajectory", fit)
    monkeypatch.setattr(agent, "hypothesis_loop", AsyncMock(return_value={
        "candidate": "mixed_drag", "rationale": "Test both drag terms"}))
    _, best, record = select(t, y)
    assert record["fallback_used"] is True
    assert best.name == record["baseline_fit"]["name"]


def test_unidentifiable_optional_candidate_cannot_win(monkeypatch):
    t, y = drag_data()
    real_fit = agent.fit_trajectory
    def fit(*args, **kwargs):
        results = real_fit(*args, **kwargs)
        return [replace(r, locally_identifiable=False) for r in results] if kwargs.get("candidate") == "mixed_drag" else results
    monkeypatch.setattr(agent, "fit_trajectory", fit)
    monkeypatch.setattr(agent, "hypothesis_loop", AsyncMock(return_value={
        "candidate": "mixed_drag", "rationale": "Test mixed drag"}))
    _, winner, record = select(t, y)
    assert record['fallback_used']
    assert record['proposal_fit']['rejection_reason'] == 'unidentifiable_structure'
    assert record['wonder_proposal']['failure_stage'] == 'numerical_validation'
    assert winner.name == record['baseline_fit']['name']
    assert 'mixed_drag' in record['bic_verdict']['evaluated_hypotheses']
    assert 'mixed_drag' not in record['bic_verdict']['eligible_hypotheses']


def test_only_simple_fit_precedes_proposal(monkeypatch):
    t, y = drag_data()
    events = []
    real_fit = agent.fit_trajectory
    def fit(*args, **kwargs):
        events.append(kwargs.get("candidate", "deterministic_library"))
        return real_fit(*args, **kwargs)
    async def propose(evidence, provider):
        assert events == ["free_fall"]
        events.append("proposal")
        return {"candidate": "mixed_drag", "rationale": "Training residuals"}
    monkeypatch.setattr(agent, "fit_trajectory", fit)
    monkeypatch.setattr(agent, "hypothesis_loop", propose)
    select(t, y)
    assert events == ["free_fall", "proposal", "deterministic_library", "mixed_drag"]
