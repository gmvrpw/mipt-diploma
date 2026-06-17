import worker_service_proto.task as proto

from src.app import (
    AddResponse,
    AnimateResponse,
    DeleteResponse,
    NormMapResponse,
    RmbgResponse,
    StandResponse,
)
from src.controller.task.kafka.mapping import (
    domain_error_to_code,
    from_add_response,
    from_animate_response,
    from_delete_response,
    from_norm_map_response,
    from_rmbg_response,
    from_stand_response,
    to_add_request,
    to_animate_request,
    to_cancelled_response,
    to_delete_request,
    to_failed_response,
    to_norm_map_request,
    to_rmbg_request,
    to_stand_request,
)
from src.domain.model.error import (
    EntityNotFoundError,
    InferenceError,
    ServicePermissionError,
    ServiceTimeoutError,
    ServiceUnavailableError,
)


def test_add_round_trip():
    task = proto.Add(id="t1", character_path="c", asset_path="a", prompt="p",
                     negative_prompt="np")
    req = to_add_request(task)
    assert (req.task_id, req.character_path, req.asset_path, req.prompt,
            req.negative_prompt) == ("t1", "c", "a", "p", "np")

    resp = AddResponse(task_id="t1", character_path="result.png")
    completed = from_add_response("t1", resp)
    assert completed.task_id == "t1"
    assert completed.character_path == "result.png"


def test_delete_round_trip():
    task = proto.Delete(id="t2", character_path="c", prompt="p")
    req = to_delete_request(task)
    assert req.task_id == "t2" and req.negative_prompt is None
    completed = from_delete_response("t2", DeleteResponse(task_id="t2", character_path="r"))
    assert completed.character_path == "r"


def test_stand_round_trip_with_optional_pose():
    task = proto.Stand(id="t3", character_path="c", pose_path="pose.png", prompt="p")
    req = to_stand_request(task)
    assert req.pose_path == "pose.png"
    completed = from_stand_response("t3", StandResponse(task_id="t3", character_path="r"))
    assert completed.task_id == "t3"


def test_animate_round_trip():
    task = proto.Animate(id="t4", first_frame_path="f", prompt="p", num_frames=45,
                         loop=True)
    req = to_animate_request(task)
    assert req.num_frames == 45 and req.loop is True
    completed = from_animate_response("t4", AnimateResponse(task_id="t4", frames_path="out"))
    assert completed.frames_path == "out"


def test_rmbg_round_trip():
    task = proto.Rmbg(id="t5", frames_path="x")
    req = to_rmbg_request(task)
    assert req.frames_path == "x"
    completed = from_rmbg_response("t5", RmbgResponse(task_id="t5", frames_path="o"))
    assert completed.frames_path == "o"


def test_norm_map_round_trip():
    task = proto.NormMap(id="t6", frames_path="x")
    req = to_norm_map_request(task)
    assert req.frames_path == "x"
    completed = from_norm_map_response("t6", NormMapResponse(task_id="t6", frames_path="o"))
    assert completed.frames_path == "o"


def test_cancelled_response():
    response = to_cancelled_response("task-42")
    assert response.task_id == "task-42"
    raw = bytes(response)
    parsed = proto.Cancelled.parse(raw)
    assert parsed.task_id == "task-42"


def test_failed_response():
    response = to_failed_response("task-42", "not_found", "missing")
    assert response.task_id == "task-42"
    assert response.error_code == "not_found"
    assert response.error_message == "missing"


def test_domain_error_to_code_table():
    assert domain_error_to_code(EntityNotFoundError()) == "not_found"
    assert domain_error_to_code(ServicePermissionError()) == "permission_denied"
    assert domain_error_to_code(ServiceTimeoutError()) == "timeout"
    assert domain_error_to_code(ServiceUnavailableError()) == "service_unavailable"
    assert domain_error_to_code(InferenceError()) == "inference_error"
