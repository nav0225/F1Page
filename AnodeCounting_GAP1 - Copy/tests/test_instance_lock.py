import os
import subprocess
import sys

import pytest

from instance_lock import InstanceLock, InstanceLockError

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

HOLD = """
import sys, time
sys.path.insert(0, sys.argv[2])
from instance_lock import InstanceLock
lock = InstanceLock(sys.argv[1])   # must stay referenced: the lock lives as long as the object
lock.acquire()
print("locked", flush=True)
time.sleep(30)
"""


def test_second_process_cannot_start(tmp_path):
    path = str(tmp_path / "data" / "counter.lock")
    holder = subprocess.Popen([sys.executable, "-c", HOLD, path, APP_DIR], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(InstanceLockError, match=f"pid {holder.pid}"):
            InstanceLock(path).acquire()
    finally:
        holder.kill()
        holder.wait()
    # lock is released when the owning process dies
    lock = InstanceLock(path)
    lock.acquire()
    lock.release()
