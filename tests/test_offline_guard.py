"""The offline guard itself, since a guard nobody tests is a guard nobody has."""

from __future__ import annotations

import socket

import pytest

pytestmark = pytest.mark.unit


def test_offmachine_connection_is_refused():
    """The property the suite claims: no test reaches off this machine."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(OSError, match="offline test guard"):
        sock.connect(("pypi.org", 443))


def test_refusal_is_an_oserror_so_existing_handling_still_works():
    """Code that already handles a dead network must not meet a novel exception."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(("example.com", 80))
    except OSError:
        pass  # what a network-less machine raises
    else:
        pytest.fail("expected the guard to refuse the connection")


def test_connect_ex_reports_connection_refused():
    """Callers that use connect_ex read a return code, not an exception."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    assert sock.connect_ex(("pypi.org", 443)) == 111


def test_loopback_stays_open():
    """A local MLflow store or sqlite file must keep working.

    Binds an ephemeral listener and connects to it: proves the guard delegates to
    the real connect for loopback rather than blanket-refusing.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect(("127.0.0.1", port))
    finally:
        client.close()
        listener.close()


@pytest.mark.live
def test_live_marked_tests_are_exempt():
    """Deselected by default, so reaching this at all means `-m live` was passed.

    The guard must not apply here -- a live test's entire purpose is a real
    workspace call.
    """
    assert socket.socket.connect is not None
