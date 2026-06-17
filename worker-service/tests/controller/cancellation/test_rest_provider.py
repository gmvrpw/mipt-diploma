import httpx
import pytest

from src.controller.cancellation.rest import (
    RestCancellationConfig,
    RestCancellationProvider,
)
from src.controller.cancellation.rest.config import RestCancellationAuthorization


def _handler_factory(responses: dict[str, int]):
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["last"] = request
        status = responses.get(request.url.path, 418)
        if status == -1:
            raise httpx.ConnectError("simulated")
        return httpx.Response(status)

    return handler, captured


@pytest.fixture
def patched_async_client(monkeypatch):
    """Monkeypatch RestCancellationProvider to use httpx MockTransport."""
    state: dict = {}

    def setup(responses: dict[str, int]):
        handler, captured = _handler_factory(responses)
        transport = httpx.MockTransport(handler)

        real_init = httpx.AsyncClient.__init__

        def fake_init(self, *args, **kwargs):
            kwargs["transport"] = transport
            kwargs.pop("cert", None)
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", fake_init)
        state["captured"] = captured
        return state

    return setup


async def test_200_is_cancelled(patched_async_client):
    state = patched_async_client({"/tasks/yes/cancelled": 200})
    provider = RestCancellationProvider(RestCancellationConfig(
        path="http://api/tasks/{task_id}/cancelled",
    ))
    async with provider:
        assert await provider.get_task_cancelled("yes") is True


async def test_404_is_not_cancelled(patched_async_client):
    state = patched_async_client({"/tasks/no/cancelled": 404})
    provider = RestCancellationProvider(RestCancellationConfig(
        path="http://api/tasks/{task_id}/cancelled",
    ))
    async with provider:
        assert await provider.get_task_cancelled("no") is False


async def test_500_treated_as_not_cancelled(patched_async_client):
    state = patched_async_client({"/tasks/oops/cancelled": 500})
    provider = RestCancellationProvider(RestCancellationConfig(
        path="http://api/tasks/{task_id}/cancelled",
    ))
    async with provider:
        assert await provider.get_task_cancelled("oops") is False


async def test_network_error_treated_as_not_cancelled(patched_async_client):
    state = patched_async_client({"/tasks/down/cancelled": -1})
    provider = RestCancellationProvider(RestCancellationConfig(
        path="http://api/tasks/{task_id}/cancelled",
    ))
    async with provider:
        assert await provider.get_task_cancelled("down") is False


async def test_outside_context_returns_false_without_request():
    provider = RestCancellationProvider(RestCancellationConfig(path="http://api/{task_id}"))
    assert await provider.get_task_cancelled("anything") is False


async def test_post_body_contains_task_id(patched_async_client):
    state = patched_async_client({"/tasks/x/cancelled": 200})
    provider = RestCancellationProvider(RestCancellationConfig(
        path="http://api/tasks/{task_id}/cancelled",
        method="POST",
    ))
    async with provider:
        assert await provider.get_task_cancelled("x") is True

    request = state["captured"]["last"]
    assert request.method == "POST"
    body = request.read()
    assert b'"id"' in body and b'"x"' in body


async def test_authorization_header_attached(patched_async_client):
    state = patched_async_client({"/tasks/x/cancelled": 200})
    provider = RestCancellationProvider(RestCancellationConfig(
        path="http://api/tasks/{task_id}/cancelled",
        authorization=RestCancellationAuthorization(header="X-Token: secret"),
    ))
    async with provider:
        await provider.get_task_cancelled("x")

    assert state["captured"]["last"].headers.get("x-token") == "secret"
