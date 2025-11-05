# app.py — Job & Application Dashboard (Streamlit)

import os
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List

# Adjust this import to match your project structure
# from database.database import DatabaseManager, setup_database
from src.database import DatabaseManager, setup_database

# ---------------------------
# App Config & Theming
# ---------------------------
st.set_page_config(
    page_title="Job Tracker",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional: Inline CSS for refined look (cards, subtle shadows, responsive paddings)
CARD_CSS = """
<style>
/* Card container */
.card {
  background: var(--background-color);
  border: 1px solid rgba(120, 120, 120, 0.15);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  box-shadow: 0 2px 10px rgba(0,0,0,0.04);
  transition: box-shadow 0.2s ease, transform 0.05s ease;
  margin-bottom: 0.9rem;
}
.card:hover { box-shadow: 0 6px 18px rgba(0,0,0,0.06); }

/* Badge styles */
.badge {
  display: inline-block;
  padding: 0.25rem 0.6rem;
  border-radius: 999rem;
  font-size: 0.78rem;
  border: 1px solid rgba(120, 120, 120, 0.2);
}

/* Headings and small labels */
.hint { opacity: 0.7; font-size: 0.85rem; }

/* Night-friendly separator */
hr { border: none; border-top: 1px solid rgba(120,120,120,0.2); margin: 0.7rem 0; }

/* Compact lists */
ul.compact { margin: 0.3rem 0 0.3rem 1.2rem; }
ul.compact li { margin: 0.2rem 0; }

/* Stretch modal-ish section */
.detail-section { padding-top: 0.2rem; }

/* Responsive “pill” buttons look */
.stButton>button {
  border-radius: 10px;
}
</style>
"""

st.markdown(CARD_CSS, unsafe_allow_html=True)

# Theme toggle (uses session state + rerun)
if "theme" not in st.session_state:
    st.session_state.theme = "auto"  # auto | light | dark

def theme_badge():
    label = "System" if st.session_state.theme == "auto" else st.session_state.theme.capitalize()
    return f"<span class='badge'>{label}</span>"

with st.sidebar:
    st.subheader("Appearance")
    theme_choice = st.radio(
        "Theme",
        options=["auto", "light", "dark"],
        index=["auto", "light", "dark"].index(st.session_state.theme),
        help="Switch between auto (system), light, or dark."
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()
    st.caption("Streamlit will adapt base colors; custom CSS uses semantic vars.")

# ---------------------------
# Database Connection
# ---------------------------
# Resolve database directory (adapt if needed)
DEFAULT_DB_DIR = Path(__file__).parent / "database"

with st.sidebar:
    st.subheader("Database")
    db_dir = st.text_input("DB directory", value=str(DEFAULT_DB_DIR))
    include_app = st.toggle("Ensure application tables", value=True, help="Creates app tables if missing.")
    colA, colB = st.columns(2)
    with colA:
        init_db = st.button("Initialize", help="Initialize core schema.")
    with colB:
        optimize_db = st.button("Optimize", help="CHECKPOINT / vacuum equivalent.")

# Safely init manager
dbm = setup_database(db_path=db_dir, include_applications=include_app)
if init_db:
    dbm.initialize_schema()
    if include_app:
        dbm.add_application_tables()
    st.toast("Database initialized", icon="✅")
if optimize_db:
    dbm.vacuum_database()
    st.toast("Database optimized", icon="🧹")

# ---------------------------
# Sidebar Filters
# ---------------------------
with st.sidebar:
    st.subheader("Filters")
    q = st.text_input("Search", placeholder="Title, company, or location...")
    status = st.selectbox("Status", ["any", "active", "inactive"], index=0)
    applied_filter = st.selectbox("Applied", ["any", "applied", "not applied"], index=2)
    limit = st.slider("Max results", 10, 500, 100, step=10)

# ---------------------------
# Helper Queries
# ---------------------------
def fetch_jobs() -> pd.DataFrame:
    where = []
    params: List[Any] = []

    if q:
        where.append("(title ILIKE ? OR company ILIKE ? OR location ILIKE ?)")
        like = f"%{q}%"
        params += [like, like, like]
    if status != "any":
        where.append("status = ?")
        params.append(status)
    if applied_filter != "any":
        where.append(f"is_applied = {'TRUE' if applied_filter=='applied' else 'FALSE'}")

    where_clause = " WHERE " + " AND ".join(where) if where else ""
    sql = f"""
        SELECT id, job_id, title, company, location, scraped_at, is_applied, status, hirer_name, hirer_profile_link, job_link
        FROM jobs
        {where_clause}
        ORDER BY scraped_at DESC
        LIMIT {limit}
    """
    rows = dbm.execute_query(sql, params if params else None)
    cols = ["id","job_id","title","company","location","scraped_at","is_applied","status","hirer_name","hirer_profile_link","job_link"]
    return pd.DataFrame(rows, columns=cols)

def fetch_job_detail(job_id: str) -> Optional[Dict[str, Any]]:
    sql = """
        SELECT id, job_id, title, company, location, description, job_link, hirer_name, hirer_profile_link, scraped_at, last_updated, status, is_applied
        FROM jobs WHERE job_id = ?
        LIMIT 1
    """
    rows = dbm.execute_query(sql, [job_id])
    if not rows:
        return None
    cols = ["id","job_id","title","company","location","description","job_link","hirer_name","hirer_profile_link","scraped_at","last_updated","status","is_applied"]
    return dict(zip(cols, rows[0]))

def fetch_applications_for_job(job_id: str) -> pd.DataFrame:
    sql = """
        SELECT application_id, applied_date, status, application_method, confirmation_number,
               response_received, response_date, notes
        FROM applications
        WHERE job_id = ?
        ORDER BY applied_date DESC
    """
    rows = dbm.execute_query(sql, [job_id])
    cols = ["application_id","applied_date","status","application_method","confirmation_number","response_received","response_date","notes"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

def fetch_qna_for_application(application_id: int) -> pd.DataFrame:
    sql = """
        SELECT fq.question_text, fr.response_value, fr.response_data, fr.answered_at
        FROM form_responses fr
        JOIN form_questions fq ON fq.question_id = fr.question_id
        WHERE fr.application_id = ?
        ORDER BY fr.answered_at ASC
    """
    rows = dbm.execute_query(sql, [application_id])
    cols = ["question_text","response_value","response_data","answered_at"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

# ---------------------------
# Header / KPI Row
# ---------------------------
left, mid, right = st.columns([1.4,1,1])
with left:
    st.title("Job Explorer")
    st.caption("Open any job to view a beautiful, full detail panel with description, hirer info, and application Q&A.")
with mid:
    stats = dbm.get_statistics()
    st.markdown(f"""
    <div class="card">
      <div class="hint">Total Jobs</div>
      <h3>{stats.get('total_jobs', 0)}</h3>
      <span class="hint">Companies: {stats.get('unique_companies', 0)}</span>
    </div>
    """, unsafe_allow_html=True)
with right:
    applied = stats.get("applied_jobs", 0)
    unapplied = stats.get("unapplied_jobs", 0)
    st.markdown(f"""
    <div class="card">
      <div class="hint">Applied</div>
      <h3>{applied}</h3>
      <span class="hint">Unapplied: {unapplied}</span>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Jobs Table + Quick Open
# ---------------------------
jobs_df = fetch_jobs()

# Unique, compact view with quick-open radio on the side
st.subheader("Results")
st.caption("Tip: Use search and filters in the left sidebar to narrow down quickly.")

if jobs_df.empty:
    st.info("No jobs found for the current filters.")
else:
    # Build a light table for overview
    show_cols = ["title","company","location","scraped_at","status","is_applied"]
    grid = jobs_df[["job_id"] + show_cols].copy()
    grid.rename(columns={
        "scraped_at": "scraped",
        "is_applied": "applied"
    }, inplace=True)

    # Two columns: left table, right quick details
    tcol, dcol = st.columns([1.8, 1.2])

    with tcol:
        # Provide a selectbox to pick a job quickly
        labels = [
            f"{r.title} — {r.company or '-'} — {r.location or '-'}" for _, r in grid.iterrows()
        ]
        job_selector = st.selectbox(
            "Open a job",
            options=grid["job_id"].tolist(),
            format_func=lambda jid: labels[ grid["job_id"].tolist().index(jid) ] if jid in grid["job_id"].tolist() else jid,
            index=0
        )

        # Show compact table beneath
        st.dataframe(
            grid.set_index("job_id"),
            use_container_width=True,
            hide_index=False
        )

    with dcol:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='hint'>Theme</div>" + theme_badge(), unsafe_allow_html=True)
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.caption("Job quick facts will appear here when selected on the left.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------
    # Full Detail Drawer
    # ---------------------------
    st.markdown("### Job details")
    st.caption("Everything about the selected job, beautifully formatted:")

    job = fetch_job_detail(job_selector)
    if job is None:
        st.error("Selected job not found.")
    else:
        # Top meta row
        col1, col2, col3, col4 = st.columns([1.4,1,1,1])
        with col1:
            st.markdown(f"#### {job['title'] or 'Untitled'}")
            st.write(f"{job['company'] or '—'} · {job['location'] or '—'}")
            if job.get("job_link"):
                st.link_button("Open Listing", job["job_link"], use_container_width=False)
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='hint'>Status</div>", unsafe_allow_html=True)
            st.write(f"{job['status']}")
            st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='hint'>Applied</div>", unsafe_allow_html=True)
            st.write("Yes ✅" if job["is_applied"] else "No")
            st.markdown("</div>", unsafe_allow_html=True)
        with col4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='hint'>Timestamps</div>", unsafe_allow_html=True)
            st.write(f"Scraped: {job['scraped_at']}")
            st.write(f"Updated: {job['last_updated']}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # Two main panels: Description | Hirer & Application
        left_p, right_p = st.columns([1.4, 1])

        with left_p:
            st.markdown("#### Description")
            st.markdown("<div class='card detail-section'>", unsafe_allow_html=True)
            desc = job.get("description") or "_No description provided._"
            st.markdown(desc, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right_p:
            st.markdown("#### Hirer details")
            st.markdown("<div class='card detail-section'>", unsafe_allow_html=True)
            st.write(f"Name: {job.get('hirer_name') or '—'}")
            if job.get("hirer_profile_link"):
                st.link_button("Open Hirer Profile", job["hirer_profile_link"])
            st.markdown("</div>", unsafe_allow_html=True)

            # Applications block
            st.markdown("#### Applications")
            apps_df = fetch_applications_for_job(job["job_id"])
            st.markdown("<div class='card detail-section'>", unsafe_allow_html=True)
            if apps_df.empty:
                st.caption("No applications yet for this job.")
                mark_apply = st.button("Mark as applied", type="primary", key=f"apply_{job['job_id']}")
                if mark_apply:
                    dbm.mark_as_applied(job["job_id"])
                    st.toast("Job marked as applied", icon="📝")
                    st.rerun()
            else:
                # Show latest first
                st.dataframe(
                    apps_df.head(10),
                    use_container_width=True,
                    hide_index=True
                )
                # Pick an application to expand Q&A
                picked_app_id = st.selectbox(
                    "View application Q&A",
                    options=apps_df["application_id"].tolist(),
                    index=0 if not apps_df.empty else None
                )
                if picked_app_id is not None:
                    qna_df = fetch_qna_for_application(int(picked_app_id))
                    if qna_df.empty:
                        st.caption("No Q&A responses captured for this application.")
                    else:
                        for _, row in qna_df.iterrows():
                            with st.expander(row["question_text"]):
                                rv = row["response_value"]
                                rd = row["response_data"]
                                if rv:
                                    st.write(rv)
                                if rd:
                                    try:
                                        st.json(json.loads(rd) if isinstance(rd, str) else rd)
                                    except Exception:
                                        st.write(str(rd))
                                st.caption(f"Answered at: {row['answered_at']}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # Action Row
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            if st.button("Refresh", use_container_width=True):
                st.rerun()
        with a2:
            if not job["is_applied"]:
                if st.button("Mark as applied", type="primary", use_container_width=True, key=f"apply2_{job['job_id']}"):
                    dbm.mark_as_applied(job["job_id"])
                    st.toast("Job marked as applied", icon="✅")
                    st.rerun()
        with a3:
            if st.button("Export job → CSV", use_container_width=True):
                tmp = Path(st.experimental_get_query_params().get("_tmp_dir", [str(Path.cwd())])[0]) / f"job_{job['job_id']}.csv"
                df = pd.DataFrame([job])
                df.to_csv(tmp, index=False)
                st.toast(f"Exported to {tmp}", icon="📤")
        with a4:
            if st.button("Open company filter", use_container_width=True):
                st.session_state["theme"] = st.session_state.get("theme", "auto")  # no-op, just a UI ping
                st.toast("Use sidebar filter to refine by company or title.", icon="🔎")
