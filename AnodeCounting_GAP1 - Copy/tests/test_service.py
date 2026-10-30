import threading
import time

import pytest

from config import load_settings
from counter_service import CounterService
from persistence import EventWriter, Outbox
from tests.sim import simulate

SETTINGS = load_settings({"COUNT_DIRECTION": "up", "SOURCE": "file", "VIDEO_PATH": "x.mp4",
                          "STARTUP_GRACE_S": "30", "HEALTH_MAX_FRAME_AGE_S": "5"}, require_sql=False)


class Script:
    """Detections for successive frames, shared by every detector instance."""

    def __init__(self, frames, fail_at=()):
        self.frames = frames
        self.fail_at = set(fail_at)
        self.i = 0
        self.lock = threading.Lock()

    def detector(self):
        def detect(frame):
            with self.lock:
                i = self.i
                self.i += 1
            if i in self.fail_at:
                raise RuntimeError(f"CUDA error at frame {i}")
            return self.frames[i] if i < len(self.frames) else []
        return detect

    @property
    def done(self):
        return self.i >= len(self.frames)


class EndlessSource:
    def read(self):
        time.sleep(0.0005)
        return True, object()

    def release(self):
        pass


class SwitchableSink:
    def __init__(self, healthy=True):
        self.healthy = healthy
        self.written = []

    def write(self, kind, payload):
        if not self.healthy:
            raise ConnectionError("SQL Server unreachable")
        self.written.append((kind, payload["uid"]))


def run_until(predicate, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def parts(tmp_path):
    outbox = Outbox(str(tmp_path / "outbox.sqlite3"))
    sink = SwitchableSink()
    writer = EventWriter(outbox, sink, min_backoff_s=0.01, max_backoff_s=0.05, idle_wait_s=0.01)
    return outbox, sink, writer


def make_service(parts, script):
    outbox, _, writer = parts
    svc = CounterService(SETTINGS, outbox, writer, script.detector, EndlessSource)
    svc.RESTART_BACKOFF_S = 0.01
    return svc


def test_counts_and_queues_events(parts):
    outbox, sink, writer = parts
    script = Script(simulate(n_anodes=3, speed=6.0))
    svc = make_service(parts, script)
    writer.start()
    svc.start()
    assert run_until(lambda: script.done)
    assert run_until(lambda: len(sink.written) == 15)
    svc.stop()
    writer.stop()
    kinds = [k for k, _ in sink.written]
    assert kinds.count("hole") == 12 and kinds.count("anode") == 3
    assert svc.stats()["anode_count"] == 3


def test_pipeline_exception_restarts_instead_of_killing_counter(parts):
    outbox, sink, writer = parts
    # fail early, before the first hole reaches the line
    script = Script(simulate(n_anodes=3, speed=6.0), fail_at={3})
    svc = make_service(parts, script)
    writer.start()
    svc.start()
    assert run_until(lambda: script.done)
    assert run_until(lambda: len(sink.written) == 15)
    assert svc.loop_restarts == 1
    assert svc.is_running()
    assert "CUDA error" in (svc.last_error or "")
    assert svc.health()["status"] == "degraded"
    svc.stop()
    writer.stop()


def test_sql_outage_does_not_block_counting_and_loses_nothing(parts):
    outbox, sink, writer = parts
    sink.healthy = False
    script = Script(simulate(n_anodes=3, speed=6.0))
    svc = make_service(parts, script)
    writer.start()
    svc.start()
    assert run_until(lambda: script.done)       # counting kept going during the outage
    assert svc.stats()["anode_count"] == 3
    assert sink.written == []
    assert outbox.backlog() == 15
    health = svc.health()
    assert health["status"] == "degraded"
    assert health["sql"]["consecutive_failures"] > 0

    sink.healthy = True                         # database comes back
    assert run_until(lambda: outbox.backlog() == 0)
    uids = [u for _, u in sink.written]
    assert len(uids) == 15 and len(set(uids)) == 15
    svc.stop()
    writer.stop()


def test_health_reports_down_when_counter_not_running(parts):
    outbox, sink, writer = parts
    svc = make_service(parts, Script([]))
    writer.start()
    report = svc.health()
    assert report["status"] == "down"
    assert "counting thread is not running" in report["problems"]
    writer.stop()


def test_health_endpoint_status_codes(parts):
    from fastapi.testclient import TestClient
    from api import create_app

    outbox, sink, writer = parts
    script = Script(simulate(n_anodes=1, speed=6.0))
    svc = make_service(parts, script)
    app = create_app(svc, writer)
    with TestClient(app) as client:
        assert run_until(lambda: svc.frames_processed > 5)
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"
        assert client.get("/stats").json()["camera"] == SETTINGS.camera_name
    # lifespan shutdown stopped both threads
    assert not svc.is_running() and not writer.is_alive()
    r = TestClient(app).get("/health")
    assert r.status_code == 503
