import time

from persistence import EventWriter, Outbox, PermanentEventError


class RecordingSink:
    """Fails the first `fail_times` writes with a transient error, then records uids."""

    def __init__(self, fail_times=0, permanent_uids=()):
        self.fail_times = fail_times
        self.permanent_uids = set(permanent_uids)
        self.written = []
        self.calls = 0

    def write(self, kind, payload):
        self.calls += 1
        if payload["uid"] in self.permanent_uids:
            raise PermanentEventError("value out of range")
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("SQL Server unreachable")
        self.written.append(payload["uid"])


def make_events(outbox, n, kind="hole"):
    return [outbox.append(kind, {"uid": f"uid-{i:03d}", "n": i}) for i in range(n)]


def test_outbox_is_fifo_and_durable_across_reopen(tmp_path):
    path = str(tmp_path / "outbox.sqlite3")
    ob = Outbox(path)
    uids = make_events(ob, 5)
    ob.mark_sent(uids[0])
    ob.close()

    reopened = Outbox(path)
    assert [e.uid for e in reopened.pending()] == uids[1:]
    assert reopened.backlog() == 4


def test_append_same_uid_twice_is_ignored(tmp_path):
    ob = Outbox(str(tmp_path / "o.sqlite3"))
    ob.append("hole", {"uid": "a"})
    ob.append("hole", {"uid": "a"})
    assert ob.backlog() == 1


def test_transient_failures_retry_in_order_without_loss_or_duplicates(tmp_path):
    ob = Outbox(str(tmp_path / "o.sqlite3"))
    uids = make_events(ob, 10)
    sink = RecordingSink(fail_times=3)
    writer = EventWriter(ob, sink, min_backoff_s=0.0)

    for _ in range(20):
        writer.drain_once()
    assert sink.written == uids          # all delivered, in order, exactly once
    assert ob.backlog() == 0
    assert writer.consecutive_failures == 0
    assert writer.written_total == 10


def test_transient_failure_blocks_queue_rather_than_skipping(tmp_path):
    ob = Outbox(str(tmp_path / "o.sqlite3"))
    make_events(ob, 3)
    sink = RecordingSink(fail_times=100)
    writer = EventWriter(ob, sink)
    writer.drain_once()
    assert sink.written == []
    assert ob.backlog() == 3
    assert writer.consecutive_failures == 1
    assert "unreachable" in writer.stats()["last_error"]


def test_permanent_error_goes_to_dead_letters_after_retries(tmp_path):
    ob = Outbox(str(tmp_path / "o.sqlite3"), max_attempts_permanent=3)
    uids = make_events(ob, 4)
    sink = RecordingSink(permanent_uids={uids[1]})
    writer = EventWriter(ob, sink)

    for _ in range(10):
        writer.drain_once()
    assert sink.written == [uids[0], uids[2], uids[3]]
    assert ob.dead_count() == 1
    assert ob.backlog() == 0


def test_writer_thread_delivers_and_stops(tmp_path):
    ob = Outbox(str(tmp_path / "o.sqlite3"))
    sink = RecordingSink(fail_times=1)
    writer = EventWriter(ob, sink, min_backoff_s=0.01, idle_wait_s=0.01)
    writer.start()
    uids = make_events(ob, 5)
    writer.notify()
    deadline = time.time() + 5
    while time.time() < deadline and len(sink.written) < 5:
        time.sleep(0.01)
    writer.stop()
    assert sink.written == uids
    assert not writer.is_alive()


def test_periodic_task_failure_does_not_stop_event_delivery(tmp_path):
    ob = Outbox(str(tmp_path / "o.sqlite3"))
    sink = RecordingSink()

    def broken_periodic():
        raise RuntimeError("daily upsert failed")

    writer = EventWriter(ob, sink, periodic=broken_periodic, periodic_every_s=0.0,
                         min_backoff_s=0.01, idle_wait_s=0.01)
    writer.start()
    uids = make_events(ob, 3)
    writer.notify()
    deadline = time.time() + 5
    while time.time() < deadline and len(sink.written) < 3:
        time.sleep(0.01)
    writer.stop()
    assert sink.written == uids


def test_count_by_date(tmp_path):
    ob = Outbox(str(tmp_path / "o.sqlite3"))
    ob.append("hole", {"uid": "a"}, local_date="2026-10-30")
    ob.append("hole", {"uid": "b"}, local_date="2026-10-30")
    ob.append("anode", {"uid": "c"}, local_date="2026-10-30")
    ob.append("hole", {"uid": "d"}, local_date="2026-10-31")
    ob.mark_sent("a")
    assert ob.count_by_date("2026-10-30") == {"hole": 2, "anode": 1}
    assert ob.count_by_date("2026-10-30", unsent_only=True) == {"hole": 1, "anode": 1}
    assert ob.count_by_date("2026-11-01") == {}


def test_outbox_file_from_previous_version_is_upgraded(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.sqlite3")
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE outbox (uid TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
                    created_utc TEXT NOT NULL, sent_utc TEXT, dead INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT)""")
    conn.execute("INSERT INTO outbox(uid, kind, payload, created_utc) VALUES ('x', 'hole', '{}', '2026-10-30')")
    conn.commit()
    conn.close()
    ob = Outbox(path)
    assert ob.backlog() == 1
    ob.append("hole", {"uid": "y"}, local_date="2026-10-30")
    assert ob.count_by_date("2026-10-30") == {"hole": 1}


def test_dates_sent_since_lists_recently_delivered_days(tmp_path):
    outbox = Outbox(str(tmp_path / "outbox.sqlite3"))
    sent = [outbox.append("hole", {"uid": f"u{i}"}, local_date=d) for i, d in
            enumerate(["2026-10-29", "2026-10-30", "2026-10-30"])]
    outbox.append("hole", {"uid": "pending"}, local_date="2026-10-31")
    for uid in sent:
        outbox.mark_sent(uid)
    assert outbox.dates_sent_since(hours=24) == ["2026-10-29", "2026-10-30"]   # undelivered day not listed
    with outbox._lock:
        outbox._conn.execute("UPDATE outbox SET sent_utc = '2026-01-01T00:00:00.000+00:00'")
    assert outbox.dates_sent_since(hours=24) == []
