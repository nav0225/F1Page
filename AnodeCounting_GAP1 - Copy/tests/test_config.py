import pytest

from config import ConfigError, load_settings

BASE = {
    "COUNT_DIRECTION": "up",
    "RTSP_URL": "rtsp://user:secret@10.0.0.1:554/Streaming/Channels/101",
    "SQL_SERVER": "localhost",
    "SQL_DB": "AnodeCounting",
    "SQL_USER": "svc",
    "SQL_PASS": "pw",
}


def test_minimal_valid_env_uses_documented_defaults():
    s = load_settings(BASE)
    assert s.count_direction == "up"
    assert s.conf_thres == 0.10            # tracker gets low-confidence detections (ByteTrack design)
    assert s.count_min_conf == 0.80
    assert s.count_evidence == "track" and s.count_min_hits == 2
    assert s.line_y == 520
    assert s.source == "rtsp"
    assert s.holes_per_anode == 4


def test_count_direction_is_required():
    env = dict(BASE)
    del env["COUNT_DIRECTION"]
    with pytest.raises(ConfigError, match="COUNT_DIRECTION"):
        load_settings(env)


def test_all_errors_reported_together():
    env = dict(BASE, CONF_THRES="abc", LINE_X1="2000", LINE_X2="10", COUNT_DIRECTION="sideways")
    with pytest.raises(ConfigError) as e:
        load_settings(env)
    msg = str(e.value)
    assert "CONF_THRES" in msg and "LINE_X1" in msg and "COUNT_DIRECTION" in msg


def test_out_of_range_threshold_rejected():
    with pytest.raises(ConfigError, match="COUNT_MIN_CONF"):
        load_settings(dict(BASE, COUNT_MIN_CONF="1.5"))


def test_video_path_only_used_when_source_is_file():
    # VIDEO_PATH alone must not switch production away from RTSP
    s = load_settings(dict(BASE, VIDEO_PATH="Video1.mp4"))
    assert s.source == "rtsp"
    s = load_settings(dict(BASE, SOURCE="file", VIDEO_PATH="Video1.mp4"))
    assert s.source == "file"
    with pytest.raises(ConfigError, match="VIDEO_PATH"):
        load_settings(dict(BASE, SOURCE="file"))


def test_sql_optional_for_offline_tools():
    env = {"COUNT_DIRECTION": "up", "SOURCE": "file", "VIDEO_PATH": "x.mp4"}
    with pytest.raises(ConfigError, match="SQL_SERVER"):
        load_settings(env)
    assert load_settings(env, require_sql=False).source == "file"


def test_trusted_connection_does_not_need_user():
    env = dict(BASE, SQL_TRUSTED="true")
    del env["SQL_USER"]
    assert load_settings(env).sql_trusted is True


def test_secrets_not_in_repr():
    text = repr(load_settings(BASE))
    assert "secret" not in text
    assert "'pw'" not in text


def test_legacy_counting_mode_selectable():
    s = load_settings(dict(BASE, COUNT_EVIDENCE="frame", CONF_THRES="0.35"))
    assert s.count_evidence == "frame" and s.conf_thres == 0.35
    with pytest.raises(ConfigError, match="COUNT_EVIDENCE"):
        load_settings(dict(BASE, COUNT_EVIDENCE="maybe"))


def test_anode_grouping_settings():
    s = load_settings(BASE)
    assert s.anode_grouping == "modulo" and s.hole_spacing_px is None and s.anode_gap_slots == 1
    s = load_settings(dict(BASE, ANODE_GROUPING="structural", HOLE_SPACING_PX="64.9", HOLE_SPACING_2_PX="108.5"))
    assert s.anode_grouping == "structural" and (s.hole_spacing_px, s.hole_spacing_2_px) == (64.9, 108.5)
    with pytest.raises(ConfigError, match="ANODE_GROUPING"):
        load_settings(dict(BASE, ANODE_GROUPING="every4th"))
