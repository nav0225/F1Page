import os
import pandas as pd
import streamlit as st
from datetime import date
from dotenv import load_dotenv
import pyodbc

load_dotenv()

st.set_page_config(page_title="Anode Counter Dashboard", layout="wide")
st.title("Anode Counting Dashboard")

camera = st.text_input("Camera Name", value=os.getenv("CAMERA_NAME", "Camera_01"))
sel_date = st.date_input("Date", value=date.today())

API_BASE = st.text_input("Edge Service URL", value="http://<edge_pc_ip>:8000")

def get_conn():
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
    return pyodbc.connect(cs)

conn = get_conn()

q1 = """
SELECT CountDate, CameraName, HoleCount, AnodeCount, LastEventTime, UpdatedAt
FROM dbo.AnodeCountDaily
WHERE CountDate = ? AND CameraName = ?
"""
df = pd.read_sql(q1, conn, params=[sel_date, camera])

c1, c2, c3 = st.columns(3)
if df.empty:
    c1.metric("Holes", 0)
    c2.metric("Anodes", 0)
    c3.write("No record found for selected date/camera.")
else:
    row = df.iloc[0]
    c1.metric("Holes", int(row["HoleCount"]))
    c2.metric("Anodes", int(row["AnodeCount"]))
    c3.write(f"Last Event: {row['LastEventTime']}\n\nUpdated: {row['UpdatedAt']}")

st.divider()

st.subheader("Live View (Annotated)")
st.markdown(f"**Stream:** {API_BASE}/mjpeg")
st.image(f"{API_BASE}/mjpeg")

st.divider()
st.subheader("Hourly Hole Crossings (from events)")

q2 = """
SELECT DATEPART(HOUR, EventTime) AS Hr, COUNT(*) AS HoleCrossings
FROM dbo.HoleCrossingEvents
WHERE CAST(EventTime AS DATE) = ? AND CameraName = ?
GROUP BY DATEPART(HOUR, EventTime)
ORDER BY Hr
"""
ev = pd.read_sql(q2, conn, params=[sel_date, camera])
conn.close()

if not ev.empty:
    st.bar_chart(ev.set_index("Hr")[["HoleCrossings"]])
else:
    st.info("No events found for this date/camera (or event logging disabled).")