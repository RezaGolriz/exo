import asyncio
import os
from contextlib import suppress

import pytest
from _pytest.capture import CaptureFixture
from exo_rs import FromSwarm, NetworkingHandle, Pidfile


@pytest.mark.asyncio
async def test_sleep_on_multiple_items() -> None:
    print("PYTHON: starting handle")
    h = NetworkingHandle.new(os.urandom(16).hex().lstrip("0"), "exo-test", 52414, 52413)
    print("PYTHON: handle started")

    recv_task = asyncio.create_task(_await_recv(h))
    try:
        # sleep for 4 ticks
        for _i in range(10):
            await asyncio.sleep(1)

            await h.gossipsub_publish("topic", b"somehting or other")
    finally:
        recv_task.cancel()
        with suppress(asyncio.CancelledError):
            await recv_task


def test_pidfile(capsys: CaptureFixture[str]):
    with capsys.disabled():
        print("\nbefore python")
        scoped_lock_file()
        print("after python")


async def _await_recv(h: NetworkingHandle):
    while True:
        event = await h.recv()
        match event:
            case FromSwarm.Connection() as c:
                print(f"PYTHON: connection update: {c}")
            case FromSwarm.Message() as m:
                print(f"PYTHON: message: {m}")


def scoped_lock_file():
    _pidfile = Pidfile("/tmp/lock.pid", 0o0600)


if __name__ == "__main__":
    asyncio.run(test_sleep_on_multiple_items())
