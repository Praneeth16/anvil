"""Enforce the offline boundary the test suite claims.

The suite's stated invariant is that everything passes with the wifi off, with
anything needing a workspace marked ``live`` and deselected by default. Until
now that was an assertion rather than a property: `-m 'not live'` filters on
markers, so an *unmarked* test making a real call would pass CI, quietly become
load-bearing, and only fail once someone ran the suite on a plane or a runner
without egress.

So the boundary is enforced where it is claimed. Any test that is not marked
``live`` gets a socket layer that refuses connections to anything off-machine.

The refusal is an :class:`OSError`, which is what a network-less machine
actually raises -- code that already handles connection failure keeps handling
it, rather than seeing a novel exception type it has no path for. A test that
genuinely needs the network is telling us it is a ``live`` test.

Loopback stays open: the suite uses a local MLflow file store and sqlite, and
blocking localhost would break those without protecting anything, since a
loopback connection cannot reach a workspace.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost", "0.0.0.0"})

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _is_loopback(address: Any) -> bool:
    """True for AF_UNIX paths and loopback TCP/UDP addresses.

    AF_UNIX addresses are plain strings and never leave the machine. TCP/UDP
    addresses are ``(host, port)`` tuples; anything whose host is not loopback is
    treated as off-machine, including a hostname that would resolve to loopback,
    because resolving it to find out would itself be a network call.
    """
    if isinstance(address, str):
        return True
    if isinstance(address, tuple) and address:
        return str(address[0]) in _LOOPBACK_HOSTS
    return False


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Refuse off-machine connections for every test not marked ``live``."""
    if request.node.get_closest_marker("live"):
        yield
        return

    def _blocked_connect(self: socket.socket, address: Any) -> None:
        if _is_loopback(address):
            _real_connect(self, address)
            return
        raise OSError(
            f"offline test guard: refused connection to {address!r}. The suite must pass "
            f"with no network. If this test genuinely needs a workspace, mark it "
            f"`@pytest.mark.live` -- it will then be deselected by default and run only "
            f"in the credentialed workflow."
        )

    def _blocked_connect_ex(self: socket.socket, address: Any) -> int:
        if _is_loopback(address):
            return int(_real_connect_ex(self, address))
        return 111  # ECONNREFUSED, as a network-less machine would report

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked_connect_ex)
    yield
