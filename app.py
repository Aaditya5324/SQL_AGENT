"""
Streamlit frontend for the SQL Query Agent.

Lets you upload a SQLite database, ask questions about it in plain
English, and see the agent's answer (plus the schema and query history).

Run:
    streamlit run app.py

Requires a HuggingFace API token (set HUGGINGFACEHUB_API_TOKEN in a .env
file, as an environment variable, or paste it into the sidebar).
"""

import os
import sqlite3
import tempfile
from urllib.parse import quote

import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

st.set_page_config(page_title="SQL Query Agent", page_icon="🗄️", layout="wide")


def create_demo_database(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            country TEXT,
            created_at DATE DEFAULT CURRENT_DATE
        );
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL NOT NULL,
            stock INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            product_id INTEGER REFERENCES products(id),
            quantity INTEGER NOT NULL,
            total REAL NOT NULL,
            order_date DATE DEFAULT CURRENT_DATE
        );
        INSERT OR IGNORE INTO customers VALUES
            (1,'Alice Johnson','alice@example.com','USA','2024-01-15'),
            (2,'Bob Smith','bob@example.com','UK','2024-02-20'),
            (3,'Carlos Lima','carlos@example.com','Brazil','2024-03-10'),
            (4,'Diana Prince','diana@example.com','USA','2024-01-05');
        INSERT OR IGNORE INTO products VALUES
            (1,'Laptop Pro','Electronics',1299.99,45),
            (2,'Wireless Mouse','Electronics',29.99,200),
            (3,'Python Book','Books',49.99,120),
            (4,'Standing Desk','Furniture',599.99,15);
        INSERT OR IGNORE INTO orders VALUES
            (1,1,1,1,1299.99,'2024-04-01'),
            (2,1,2,2,59.98,'2024-04-01'),
            (3,2,3,1,49.99,'2024-04-05'),
            (4,3,4,1,599.99,'2024-04-10'),
            (5,4,1,1,1299.99,'2024-04-12'),
            (6,2,2,3,89.97,'2024-04-15');
    """)
    conn.commit()
    conn.close()


def sqlite_uri(db_path: str, read_only: bool = True) -> str:
    abs_path = os.path.abspath(db_path)
    if read_only:
        return f"sqlite:///file:{quote(abs_path)}?mode=ro&uri=true"
    return f"sqlite:///{abs_path}"


@st.cache_resource(show_spinner=False)
def build_agent(db_path: str, read_only: bool, hf_token: str):
    """Cached so the agent isn't rebuilt on every rerun (Streamlit reruns
    the whole script on each interaction). Cache key = args, so a new
    db_path / mode / token automatically builds a fresh agent."""
    db = SQLDatabase.from_uri(sqlite_uri(db_path, read_only=read_only))

    endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        provider="auto",
        huggingfacehub_api_token=hf_token,
    )
    llm = ChatHuggingFace(llm=endpoint)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )
    return agent, db


# --------------------------------------------------------------------------
# Sidebar — data source + credentials
# --------------------------------------------------------------------------
st.sidebar.header("1. Connect a database")

source = st.sidebar.radio("Source", ["Upload a .sqlite file", "Use demo database"], label_visibility="collapsed")

uploaded_file = None
if source == "Upload a .sqlite file":
    uploaded_file = st.sidebar.file_uploader(
        "SQLite database", type=["sqlite", "db", "sqlite3"], label_visibility="collapsed"
    )

allow_write = st.sidebar.checkbox("Allow write (read-write mode)", value=False,
                                   help="Off = read-only connection. On = agent can INSERT/UPDATE/DELETE.")
if allow_write:
    st.sidebar.caption("⚠️ The agent can modify this database. Only enable on data you're OK losing.")


def _default_token() -> str:
    """Check Streamlit secrets first (used on Streamlit Cloud / HF Spaces),
    then fall back to a local environment variable / .env file."""
    try:
        return st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
    except Exception:
        return os.getenv("HUGGINGFACEHUB_API_TOKEN", "")


_preset_token = _default_token()

st.sidebar.header("2. HuggingFace API token")
if _preset_token:
    st.sidebar.success("Using token from server configuration.")
    hf_token = _preset_token
else:
    hf_token = st.sidebar.text_input(
        "Token",
        type="password",
        value="",
        help="Needed to call the Llama-3.1-8B-Instruct model. Get one at huggingface.co/settings/tokens",
        label_visibility="collapsed",
    )

st.sidebar.divider()
if st.sidebar.button("Reset session"):
    st.session_state.clear()
    st.cache_resource.clear()
    st.rerun()


# --------------------------------------------------------------------------
# Resolve the database path to use this run (single source of truth)
# --------------------------------------------------------------------------
db_path = None
db_label = None

if source == "Use demo database":
    demo_path = os.path.join(tempfile.gettempdir(), "sql_agent_demo.sqlite")
    if not os.path.exists(demo_path):
        create_demo_database(demo_path)
    db_path, db_label = demo_path, "demo.sqlite (e-commerce sample data)"
elif uploaded_file is not None:
    if st.session_state.get("uploaded_name") != uploaded_file.name:
        tmp_dir = tempfile.mkdtemp()
        saved_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_path = saved_path
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.history = []
    db_path, db_label = st.session_state.uploaded_path, uploaded_file.name


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("🗄️ Natural language SQL query agent")
st.caption("Upload a SQLite database, ask questions in plain English, get grounded answers.")

if db_path is None:
    st.info("⬅️ Upload a `.sqlite` file or select the demo database from the sidebar to get started.")
    st.stop()

if not hf_token:
    st.warning("⬅️ Enter your HuggingFace API token in the sidebar to continue.")
    st.stop()

try:
    with st.spinner("Connecting and preparing the agent..."):
        agent, db = build_agent(db_path, not allow_write, hf_token)
except Exception as e:
    st.error(f"Couldn't connect to the database or build the agent:\n\n{e}")
    st.stop()

st.success(f"Connected to **{db_label}** — mode: {'read-write' if allow_write else 'read-only'}")

if "history" not in st.session_state:
    st.session_state.history = []

tab_ask, tab_schema = st.tabs(["💬 Ask", "📋 Schema"])

with tab_schema:
    st.write(f"**Tables:** {', '.join(db.get_table_names())}")
    st.code(db.get_table_info(), language="sql")

with tab_ask:
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question about your data",
            placeholder="e.g. Which customer has spent the most in total?",
        )
        submitted = st.form_submit_button("Run query", type="primary")

    if submitted and question.strip():
        with st.spinner("Agent is thinking (this can take a few tool calls)..."):
            try:
                result = agent.invoke({"input": question})
                answer = result["output"]
                error = None
            except Exception as e:
                answer = None
                error = str(e)
        st.session_state.history.insert(0, {"question": question, "answer": answer, "error": error})

    if not st.session_state.history:
        st.caption("No questions asked yet. Try: \"How many orders were placed in April 2024?\"")

    for item in st.session_state.history:
        with st.container(border=True):
            st.markdown(f"**Q:** {item['question']}")
            if item["error"]:
                st.error(f"Error: {item['error']}")
            else:
                st.markdown(f"**A:** {item['answer']}")