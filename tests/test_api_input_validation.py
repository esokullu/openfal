"""Invalid generation requests return 422 before any job/model execution."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from openfal.config import EngineConfig
from openfal.server.app import create_app


@pytest.fixture
def client_and_submit(tmp_path):
    app = create_app(EngineConfig(
        weights_dir=tmp_path / "weights", output_dir=tmp_path / "outputs", cache_dir=tmp_path / "cache",
    ))
    job = SimpleNamespace(
        status="COMPLETED", result={"images": []},
        to_status_dict=lambda **kwargs: {"status": "IN_QUEUE", "request_id": "test-request"},
    )
    submit = Mock(return_value=job)
    app.state.queue_mgr.submit_job = submit
    with TestClient(app) as client:
        yield client, submit


@pytest.fixture(params=[
    ("/minimax/h3/text-to-image", {}),
    ("/queue/minimax/h3/text-to-image", {}),
    ("/minimax/h3/text-to-image?queue=true", {}),
    ("/minimax/h3/text-to-image", {"x-fal-queue": "true"}),
])
def submission_route(request):
    return request.param


@pytest.mark.parametrize("raw,location,error_type", [
    ('{"duration": 0}', ["body", "duration"], "greater_than_equal"),
    ('{"input": {"resolution": "invalid"}}', ["body", "input", "resolution"], "enum"),
    ('{"duration": 0, "input": {"prompt": "valid"}}', ["body", "duration"], "greater_than_equal"),
    ('{"input": []}', ["body", "input"], "dict_type"),
    ('{"input": null}', ["body", "input"], "dict_type"),
    ('[]', ["body"], "dict_type"),
    ('"prompt"', ["body"], "dict_type"),
    ('42', ["body"], "dict_type"),
    ('null', ["body"], "missing"),
    ('{"prompt":', None, "json_invalid"),
])
def test_invalid_generation_body_returns_422_without_submission(
    client_and_submit, submission_route, raw, location, error_type,
):
    client, submit = client_and_submit
    route, headers = submission_route
    response = client.post(route, content=raw, headers={"Content-Type": "application/json", **headers})
    assert response.status_code == 422, response.text
    error = response.json()["detail"][0]
    assert error["type"] == error_type
    if location is not None:
        assert error["loc"] == location
    submit.assert_not_called()


@pytest.mark.parametrize("wrapped", [False, True])
def test_valid_flat_and_wrapped_inputs_preserve_payload(client_and_submit, submission_route, wrapped):
    client, submit = client_and_submit
    route, headers = submission_route
    fields = {"prompt": "a poodle", "duration": 2, "seed": 0, "custom_option": "preserved"}
    body = {"prompt": "overridden", "input": fields} if wrapped else fields
    response = client.post(route, json=body, headers=headers)
    assert response.status_code == 200, response.text
    submit.assert_called_once()
    actual = submit.call_args.kwargs["payload"]
    for key, value in fields.items():
        assert actual[key] == value
    assert actual["num_inference_steps"] is None
    assert "input" not in actual
