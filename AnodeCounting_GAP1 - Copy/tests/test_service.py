import threading
import time

import pytest

from capture import CapturedFrame, FrameGrabber
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
    def __init__(self, clock=None):
        self.clock = clock

    def read(self):
        time.sleep(0.0005)
        if self.clock is not None:
            self.clock.tick()
        return True, object()

    def release(self):
        pass


def file_grabber(clock=None):
    """Every frame in order (like a video file); optional test clock ticks per frame read."""
    return FrameGrabber(lambda: EndlessSource(clock), drop_stale=False, clock=clock)


class SwitchableSink:
    def __init__(self, healthy=True):
        self.healthy = healthy
        self.written = []

    def write(self, kind, payload):
        if not self.healthy:
            raise ConnectionError("SQL Server unreachable")
        self.written.append((kind, payload["uid"]))


def kinds(sink):
    """Number of delivered events per kind."""
    out = {}
    for kind, _ in sink.written:
        out[kind] = out.get(kind, 0) + 1
    return out


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
    svc = CounterService(SETTINGS, outbox, writer, script.detector, file_grabber())
    svc.RESTART_BACKOFF_S = 0.01
    return svc


def test_counts_and_queues_events(parts):
    outbox, sink, writer = parts
    script = Script(simulate(n_anodes=3, speed=6.0))
    svc = make_service(parts, script)
    writer.start()
    svc.start()
    assert run_until(lambda: script.done)
    assert run_until(lambda: kinds(sink).get("anode_shadow") == 3)
    svc.stop()
    writer.stop()
    # counted: 12 holes, 3 anodes (modulo); comparison: 3 structural anodes
    assert kinds(sink) == {"hole": 12, "anode": 3, "anode_shadow": 3}
    assert svc.stats()["anode_count"] == 3
    assert svc.health()["counting"]["shadow_grouping"]["status"] == {"CONFIRMED": 3}


def test_pipeline_exception_restarts_instead_of_killing_counter(parts):
    outbox, sink, writer = parts
    # fail early, before the first hole reaches the line
    script = Script(simulate(n_anodes=3, speed=6.0), fail_at={3})
    svc = make_service(parts, script)
    writer.start()
    svc.start()
    assert run_until(lambda: script.done)
    assert run_until(lambda: kinds(sink).get("anode") == 3 and kinds(sink).get("hole") == 12)
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
    assert outbox.backlog() == 18                 # 12 holes, 3 anodes, 3 comparison anodes
    health = svc.health()
    assert health["status"] == "degraded"
    assert health["sql"]["consecutive_failures"] > 0

    sink.healthy = True                         # database comes back
    assert run_until(lambda: outbox.backlog() == 0)
    uids = [u for _, u in sink.written]
    assert len(uids) == 18 and len(set(uids)) == 18
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


def test_totals_roll_over_at_plant_midnight(parts):
    outbox, sink, writer = parts
    # first hole reaches the line at frame 30; holes then every 10 frames (5 s)
    # each frame read advances the clock 0.5 s; events take the frame's capture time
    clock = SteppingClock(datetime(2026, 10, 30, 23, 59, 30, tzinfo=IST), step_s=0.5)
    script = Script(simulate(n_anodes=3, speed=6.0))
    svc = CounterService(SETTINGS, outbox, writer, script.detector, file_grabber(clock),
                         clock=clock, local_tz=IST)
    writer.start()
    svc.start()
    assert run_until(lambda: script.done)
    assert run_until(lambda: outbox.backlog() == 0)
    svc.stop()
    writer.stop()

    before, after = outbox.count_by_date("2026-10-30"), outbox.count_by_date("2026-10-31")
    assert before == {"hole": 3}                      # 23:59:45.5, :50.5, :55.5
    assert after == {"hole": 9, "anode": 3, "anode_shadow": 3}   # from 00:00:00.5 onwards
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
    second = CounterService(SETTINGS, outbox, second_writer, script2.detector, file_grabber())
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


# ----------------------------------------------------------------------------
# Counting gaps (D6)
# ----------------------------------------------------------------------------
class ScriptedRealtimeGrabber:
    """Live-camera-like grabber that hands over pre-built frames with chosen timestamps."""

    realtime = True

    def __init__(self, frames, failures_at=None):
        self.frames = list(frames)
        self.failures_at = failures_at or {}   # frame seq -> capture failure count seen by then
        self.failures = 0
        self.given = 0

    def start(self):
        pass

    def stop(self):
        pass

    def is_alive(self):
        return True

    def get(self, timeout=0.5):
        if self.given >= len(self.frames):
            time.sleep(0.01)
            return None
        f = self.frames[self.given]
        self.given += 1
        self.failures = self.failures_at.get(f.seq, self.failures)
        return f

    def stats(self):
        return {"connected": True, "failures": self.failures, "frames_read": self.given,
                "frames_dropped": 0, "reconnects": 0, "capture_fps": 20.0, "last_error": None}


T0 = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)


def frames_at(points):
    """points: list of (seq, seconds after T0)"""
    return [CapturedFrame(seq, None, 1000.0 + t, T0 + timedelta(seconds=t)) for seq, t in points]


def gaps_in(outbox):
    import json, sqlite3
    rows = sqlite3.connect(outbox.path).execute(
        "SELECT payload FROM outbox WHERE kind='gap' ORDER BY rowid").fetchall()
    return [json.loads(r[0]) for r in rows]


def run_gap_scenario(parts, grabber, state_last_frame=None):
    outbox, sink, writer = parts
    if state_last_frame is not None:
        outbox.set_state("last_frame_utc", state_last_frame.isoformat())
    svc = CounterService(SETTINGS, outbox, writer, lambda: (lambda img: []), grabber, local_tz=IST)
    svc.start()
    assert run_until(lambda: grabber.given >= len(grabber.frames))
    assert run_until(lambda: svc.frames_processed == len(grabber.frames))
    svc.stop()
    return svc, gaps_in(outbox)


def test_regular_frames_record_no_gap(parts):
    frames = frames_at([(i, i * 0.05) for i in range(1, 40)])
    svc, gaps = run_gap_scenario(parts, ScriptedRealtimeGrabber(frames))
    assert gaps == [] and svc.gaps_recorded == 0


def test_skipped_frames_record_processing_gap(parts):
    frames = frames_at([(1, 0.0), (2, 0.05), (40, 2.0), (41, 2.05)])   # 37 frames skipped
    svc, gaps = run_gap_scenario(parts, ScriptedRealtimeGrabber(frames))
    assert [g["reason"] for g in gaps] == ["PROCESSING_GAP"]
    assert gaps[0]["duration_ms"] == 1950
    assert gaps[0]["start_time"] == "2026-10-30T15:30:00.050000"      # local plant time (IST)
    assert "37 frames skipped" in gaps[0]["details"]


def test_camera_silence_records_source_stall(parts):
    frames = frames_at([(1, 0.0), (2, 0.05), (3, 3.0)])
    _, gaps = run_gap_scenario(parts, ScriptedRealtimeGrabber(frames))
    assert [g["reason"] for g in gaps] == ["SOURCE_STALL"]


def test_reconnect_records_source_disconnected(parts):
    frames = frames_at([(1, 0.0), (2, 0.05), (3, 8.0), (4, 8.05)])
    _, gaps = run_gap_scenario(parts, ScriptedRealtimeGrabber(frames, failures_at={3: 4}))
    assert [g["reason"] for g in gaps] == ["SOURCE_DISCONNECTED"]
    assert gaps[0]["duration_ms"] == 7950


def test_service_downtime_recorded_at_next_start(parts):
    frames = frames_at([(1, 0.0), (2, 0.05)])
    _, gaps = run_gap_scenario(parts, ScriptedRealtimeGrabber(frames),
                               state_last_frame=T0 - timedelta(minutes=10))
    assert [g["reason"] for g in gaps] == ["SERVICE_DOWN"]
    assert gaps[0]["duration_ms"] == 600_000
    outbox = parts[0]
    assert outbox.get_state("last_frame_utc") == (T0 + timedelta(seconds=0.05)).isoformat()


def test_events_use_capture_time_not_processing_time(parts):
    outbox, sink, writer = parts
    sim_frames = simulate(n_anodes=1, speed=6.0)
    frames = frames_at([(i + 1, i * 0.05) for i in range(len(sim_frames))])
    script = Script(sim_frames)
    svc = CounterService(SETTINGS, outbox, writer, script.detector, ScriptedRealtimeGrabber(frames),
                         local_tz=IST)
    svc.start()
    assert run_until(lambda: svc.frames_processed == len(frames))
    svc.stop()
    import json, sqlite3
    first = json.loads(sqlite3.connect(outbox.path).execute(
        "SELECT payload FROM outbox WHERE kind='hole' ORDER BY rowid LIMIT 1").fetchone()[0])
    # first hole crosses at simulated frame index 30 -> captured 1.5 s after T0
    assert first["event_time_utc"] == "2026-10-30T10:00:01.500000"
    assert first["lag_ms"] >= 0


def test_preview_is_encoded_only_while_a_viewer_is_connected(parts):
    outbox, sink, writer = parts
    encodes = []
    script = Script(simulate(n_anodes=3, speed=6.0))
    svc = CounterService(SETTINGS, outbox, writer, script.detector, file_grabber(),
                         encode_preview=lambda *a: encodes.append(1) or b"jpeg")
    writer.start()
    svc.start()
    assert run_until(lambda: svc.frames_processed > 50)
    assert encodes == []                              # nobody watching: no encoding at all
    svc.preview_client(+1)
    t0 = time.time()
    assert run_until(lambda: len(encodes) >= 3)
    elapsed = time.time() - t0
    svc.preview_client(-1)
    n = len(encodes)
    time.sleep(0.5)
    svc.stop()
    writer.stop()
    assert len(encodes) <= n + 1                      # stops again when the viewer leaves
    assert n <= SETTINGS.preview_max_fps * (elapsed + 0.5) + 2   # capped at PREVIEW_MAX_FPS
    seq, jpeg = svc.preview_jpeg()
    assert jpeg == b"jpeg" and seq >= 3
