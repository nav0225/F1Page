"""
Operator dashboard for the anode counter.

Anode numbers come from dbo.AnodeEvents - the same table the Power BI report
counts - shown together with the evidence behind them: periods in which the
belt could not be observed (counting gaps), crossings rejected for low
confidence, and the grouping status of each anode. The live section reads
the counter service's /health and /stats.

    streamlit run dashboard_app.py
"""
import json
import os
import urllib.request
from datetime import date

import streamlit as st
from dotenv import load_dotenv

from dashboard_data import DashboardData, hourly, unobserved_seconds

load_dotenv()

st.set_page_config(page_title="Anode Counter Dashboard", layout="wide")
st.title("Anode Counting Dashboard")

SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:8000").rstrip("/")
REFRESH_S = int(os.getenv("DASHBOARD_REFRESH_S", "30"))

camera = st.sidebar.text_input("Camera Name", value=os.getenv("CAMERA_NAME", "Camera_01"))
sel_date = st.sidebar.date_input("Date", value=date.today())
service_url = st.sidebar.text_input("Counter service URL", value=SERVICE_URL)
show_video = st.sidebar.checkbox("Show live video", value=False,
                                 help="Streams the annotated camera view from the counter service.")


def get_conn():
    import pyodbc

    trusted = os.getenv("SQL_TRUSTED", "false").lower() == "true"
    driver = os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server")
    server = os.getenv("SQL_SERVER", "")
    db = os.getenv("SQL_DB", "")
    user = os.getenv("SQL_USER", "")
    pwd = os.getenv("SQL_PASS", "")

    if trusted:
        cs = f"DRIVER={{{driver}}};SERVER={server};DATABASE={db};Trusted_Connection=yes;TrustServerCertificate=yes;"
    else:
        cs = f"DRIVER={{{driver}}};SERVER={server};DATABASE={db};UID={user};PWD={pwd};TrustServerCertificate=yes;"
    return pyodbc.connect(cs, timeout=5)


def fetch_json(path):
    try:
        with urllib.request.urlopen(f"{service_url}{path}", timeout=2) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:   # /health answers 503 with a JSON body when down
        try:
            return json.loads(e.read().decode("utf-8"))
        except Exception:
            return None
    except Exception:
        return None


# ----------------------------
# Live service status
# ----------------------------
def live_section():
    st.subheader("Counter service (live)")
    health = fetch_json("/health")
    if health is None:
        st.error(f"Counter service not reachable at {service_url}")
        return
    status = health.get("status", "unknown")
    badge = {"ok": "🟢 OK", "degraded": "🟠 DEGRADED", "down": "🔴 DOWN"}.get(status, status)
    stats = fetch_json("/stats") or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Status", badge)
    c2.metric("Anodes today", stats.get("anode_count", "-"))
    c3.metric("Holes today", stats.get("hole_count", "-"))
    lag = (health.get("lag_ms") or {}).get("p95")
    c4.metric("Lag p95", f"{lag / 1000:.1f} s" if lag is not None else "-")
    c5.metric("Visibility", (health.get("visibility") or {}).get("state", "-"))
    for problem in health.get("problems", []):
        st.warning(problem)
    st.caption(f"Last frame {health.get('frame_age_s')} s ago · capture "
               f"{(health.get('capture') or {}).get('capture_fps', '-')} fps · SQL backlog "
               f"{(health.get('sql') or {}).get('backlog', '-')} · totals source {health.get('totals_source', '-')}")


if hasattr(st, "fragment"):
    st.fragment(run_every=REFRESH_S)(live_section)()
else:  # older Streamlit: refreshes when the page is re-run
    live_section()

if show_video:
    st.image(f"{service_url}/mjpeg", caption="Live view (annotated)")

st.divider()

# ----------------------------
# Selected day (from SQL)
# ----------------------------
st.subheader(f"Handover record - {sel_date.isoformat()}")
try:
    conn = get_conn()
except Exception as e:
    st.error(f"Database not reachable: {e}")
    st.stop()

try:
    data = DashboardData(conn, camera)
    anodes = data.anodes(sel_date)
    holes = data.hole_count(sel_date)
    gaps = data.gaps(sel_date)
    rejected = data.rejected_count(sel_date)
    shadow = data.shadow_summary(sel_date)
    daily = data.daily_row(sel_date)
finally:
    conn.close()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Anodes", len(anodes))
c2.metric("Holes", holes)
if gaps is not None:
    unobserved = unobserved_seconds(gaps, sel_date)
    c3.metric("Belt not observable", f"{unobserved / 60:.1f} min",
              help="Total time recorded in counting gaps: anodes passing then were not counted.")
else:
    c3.metric("Belt not observable", "n/a", help="Apply sql_migrations/003 to record counting gaps.")
c4.metric("Rejected crossings", "n/a" if rejected is None else rejected,
          help="Objects that crossed the line without enough confident detections; not counted.")

if "CountStatus" in anodes.columns and not anodes.empty:   # NULL = not assessed (modulo grouping)
    st.caption("Anode evidence: " + ", ".join(
        f"{k}: {v}" for k, v in anodes["CountStatus"].fillna("NOT ASSESSED").value_counts().items()))
if daily is not None and int(daily["AnodeCount"]) != len(anodes):
    st.info(f"AnodeCountDaily shows {int(daily['AnodeCount'])} anodes for this day; the event table "
            f"shows {len(anodes)}. Days before migration 002 were overwritten by the old service.")

st.markdown("**Anodes per hour**")
st.bar_chart(hourly(anodes))

if gaps is not None and not gaps.empty:
    st.markdown("**Counting gaps** - periods in which anodes could pass unobserved")
    st.dataframe(gaps, hide_index=True)

if shadow is not None and not shadow.empty:
    st.markdown("**Grouping comparison** - anodes from the grouping method that is not used for counting")
    st.dataframe(shadow, hide_index=True)
