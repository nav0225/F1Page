"""
Single-instance guard for the counter service.

Only one process may own a camera's counting state and outbox. A second one
(uvicorn --workers 2, or a test copy started by mistake on the same server)
would open its own camera stream and write every event a second time. The
lock is an OS file lock, released automatically if the process dies.
"""
import os


class InstanceLockError(RuntimeError):
    pass


class InstanceLock:
    """Keep the object referenced for the life of the process: the lock is
    released when its file handle is closed or garbage-collected."""

    def __init__(self, path: str):
        self.path = path
        self._fh = None

    def acquire(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        fh = open(self.path, "a+")
        try:
            fh.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.seek(0)
            owner = fh.read().strip() or "unknown"
            fh.close()
            raise InstanceLockError(
                f"Another counter instance (pid {owner}) is already running with {self.path}. "
                "Run exactly one counter process per camera (uvicorn with a single worker).")
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is not None:
            self._fh.close()  # closing the handle releases the OS lock
            self._fh = None
