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


# ----------------------------------------------------------------------------
# Day boundaries and restarts (D2, D3)
# ----------------------------------------------------------------------------
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))


class SteppingClock:
    """UTC clock that advances by `step` every time the detector runs a frame."""

    def __init__(self, start_local, step_s):
        self.now = start_local.astimezone(timezone.utc)
        self.step = timedelta(seconds=step_s)
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.now

    def tick(self):
        with self.lock:
            self.now += self.step


def clocked(script, clock):
    """Detector factory whose frames advance the clock."""
    def factory():
        inner = script.detector()

        def detect(frame):
            clock.tick()
            return inner(frame)
        return detect
    return factory


def test_totals_roll_over_at_plant_midnight(parts):
    outbox, sink, writer = parts
    # first hole reaches the line at frame 30; holes then every 10 frames (5 s)
    clock = SteppingClock(datetime(2026, 10, 30, 23, 59, 30, tzinfo=IST), step_s=0.5)
    script = Script(simulate(n_anodes=3, speed=6.0))
    svc = CounterService(SETTINGS, outbox, writer, clocked(script, clock), EndlessSource,
                         clock=clock, local_tz=IST)
    writer.start()
    svc.start()
    assert run_until(lambda: script.done)
    assert run_until(lambda: outbox.backlog() == 0)
    svc.stop()
    writer.stop()

    before, after = outbox.count_by_date("2026-10-30"), outbox.count_by_date("2026-10-31")
    assert before == {"hole": 3}                      # 23:59:45, :50, :55
    assert after == {"hole": 9, "anode": 3}           # from 00:00:00 onwards
    stats = svc.stats()
    assert stats["date"] == "2026-10-31"
    assert (stats["hole_count"], stats["anode_count"]) == (9, 3)

    # local and UTC timestamps describe the same instant
    import json, sqlite3
    payload = json.loads(sqlite3.connect(outbox.path).execute(
        "SELECT payload FROM outbox WHERE kind='hole' ORDER BY rowid LIMIT 1").fetchone()[0])
    local = datetime.fromisoformat(payload["event_time"]).replace(tzinfo=IST)
    utc = datetime.fromisoformat(payload["event_time_utc"]).replace(tzinfo=timezone.utc)
    assert local == utc


def test_restart_keeps_todays_totals(parts, tmp_path):
    outbox, sink, writer = parts
    script = Script(simulate(n_anodes=3, speed=6.0))
    first = make_service(parts, script)
    first.restore_today()
    writer.start()
    first.start()
    assert run_until(lambda: script.done)
    assert run_until(lambda: outbox.backlog() == 0)
    first.stop()
    writer.stop()
    assert first.stats()["anode_count"] == 3

    # Service restarts. SQL holds the 15 delivered events plus 20 holes / 5 anodes
    # recorded earlier today by the previous service version.
    second_writer = EventWriter(outbox, sink, min_backoff_s=0.01, idle_wait_s=0.01)
    script2 = Script(simulate(n_anodes=2, speed=6.0))
    second = CounterService(SETTINGS, outbox, second_writer, script2.detector, EndlessSource)
    second.RESTART_BACKOFF_S = 0.01
    second.restore_today(lambda day: (12 + 20, 3 + 5, None))
    assert (second.stats()["hole_count"], second.stats()["anode_count"]) == (32, 8)
    assert second.stats()["totals_source"] == "sql+outbox"

    second_writer.start()
    second.start()
    assert run_until(lambda: script2.done)
    assert run_until(lambda: outbox.backlog() == 0)
    second.stop()
    second_writer.stop()
    assert (second.stats()["hole_count"], second.stats()["anode_count"]) == (40, 10)


def test_restore_adds_undelivered_events_and_falls_back_without_sql(parts):
    outbox, sink, writer = parts
    svc = make_service(parts, Script([]))
    today = svc.local_today().isoformat()
    for i in range(6):
        outbox.append("hole", {"uid": f"h{i}"}, local_date=today)
    outbox.append("anode", {"uid": "a0"}, local_date=today)
    for uid in ("h0", "h1", "h2", "h3"):
        outbox.mark_sent(uid)  # already in SQL

    svc.restore_today(lambda day: (4, 0, None))      # SQL has the 4 delivered holes
    assert (svc.stats()["hole_count"], svc.stats()["anode_count"]) == (6, 1)

    def sql_down(day):
        raise ConnectionError("SQL Server unreachable")

    svc.restore_today(sql_down)
    assert (svc.stats()["hole_count"], svc.stats()["anode_count"]) == (6, 1)
    assert svc.stats()["totals_source"] == "outbox_only"
