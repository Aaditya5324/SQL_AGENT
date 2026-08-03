# 🕵️‍♂️ SQL Query Agent | Streamlit UI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Llama_3.1-F9AB00?style=flat-square&logo=huggingface&logoColor=black)

An interactive web interface for the LangChain SQL Agent. Upload your SQLite databases, ask questions in plain English, and let the AI write and execute the SQL to get answers grounded in your data.

> **Preview**  
> `![App Screenshot](image.png)`

---

## ✨ Features

*   🗣️ **Natural Language to SQL:** Ask complex questions about your data in plain English.
*   📁 **Bring Your Own Data:** Upload any `.sqlite` or `.db` file, or use the built-in e-commerce demo database.
*   🔒 **Permission Controls:** Run the agent in **Read-Only** (default) or **Read-Write** mode if you trust the LLM to run `INSERT`/`UPDATE`/`DELETE` operations.
*   🧠 **Agentic Workflow:** The LangChain agent automatically inspects your schema, lists tables, drafts SQL, and executes it.
*   🔍 **Schema Explorer:** Instantly view all tables and column definitions in a dedicated tab.

---

## 🛠️ Tech Stack

*   **Frontend:** Streamlit
*   **LLM Orchestration:** LangChain (SQL Agent)
*   **Language Model:** `meta-llama/Llama-3.1-8B-Instruct` (via Hugging Face)
*   **Database:** SQLite


