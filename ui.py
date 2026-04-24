import base64
import json
import streamlit as st
from datetime import datetime
from pathlib import Path
from scrape import scrape_multiple
from search import get_search_results
from llm_utils import BufferedStreamingHandler, get_model_choices
from llm import get_llm, refine_query, filter_results, generate_summary, PRESET_PROMPTS
from config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OLLAMA_BASE_URL,
    LLAMA_CPP_BASE_URL,
)
from health import check_llm_health, check_search_engines, check_tor_proxy


def _render_pipeline_error(stage: str, err: Exception) -> None:
    message = str(err).strip() or err.__class__.__name__
    lower_msg = message.lower()
    hints = [
        "- Confirm the relevant API key is set in your `.env` or shell before launching Streamlit.",
        "- Keys copied from dashboards often include hidden spaces; re-copy if authentication keeps failing.",
        "- Restart the app after updating environment variables so the new values are picked up.",
    ]
    if any(token in lower_msg for token in ("anthropic", "x-api-key", "invalid api key", "authentication")):
        hints.insert(0, "- Claude/Anthropic models require a valid `ANTHROPIC_API_KEY`.")
    elif "openrouter" in lower_msg or "user not found" in lower_msg or "code: 401" in lower_msg:
        hints.insert(0, "- OpenRouter 401/User not found usually means the API key is invalid/expired or has leading/trailing characters.")
        hints.insert(1, "- Set `OPENROUTER_API_KEY` without extra spaces and verify the key is active in your OpenRouter account.")
        hints.insert(2, "- Keep `OPENROUTER_BASE_URL` as `https://openrouter.ai/api/v1` unless you intentionally use a custom gateway.")
    elif "openai" in lower_msg or "gpt" in lower_msg:
        hints.insert(0, "- OpenAI models require `OPENAI_API_KEY` with access to the chosen model.")
    elif "google" in lower_msg or "gemini" in lower_msg:
        hints.insert(0, "- Google Gemini models need `GOOGLE_API_KEY` or Application Default Credentials.")
    st.error("❌ Failed to {}.\n\nError: {}\n\n{}".format(stage, message, "\n".join(hints)))
    st.stop()


# ---------------------------------------------------------------------------
# Investigation persistence
# ---------------------------------------------------------------------------

INVESTIGATIONS_DIR = Path("investigations")


def save_investigation(query, refined_query, model, preset_label, sources, summary):
    """Save a completed investigation to disk. Returns the filename."""
    INVESTIGATIONS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"investigation_{timestamp}.json"
    data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "refined_query": refined_query,
        "model": model,
        "preset": preset_label,
        "sources": sources,
        "summary": summary,
    }
    (INVESTIGATIONS_DIR / fname).write_text(json.dumps(data, indent=2))
    return fname


def load_investigations():
    """Return list of saved investigations sorted newest-first."""
    if not INVESTIGATIONS_DIR.exists():
        return []
    files = sorted(INVESTIGATIONS_DIR.glob("investigation_*.json"), reverse=True)
    investigations = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            data["_filename"] = f.name
            investigations.append(data)
        except Exception:
            continue
    return investigations


# ---------------------------------------------------------------------------
# Cached backend calls
# ---------------------------------------------------------------------------

@st.cache_data(ttl=200, show_spinner=False)
def cached_search_results(refined_query: str, threads: int):
    return get_search_results(refined_query.replace(" ", "+"), max_workers=threads)


@st.cache_data(ttl=200, show_spinner=False)
def cached_scrape_multiple(filtered: list, threads: int, max_content_chars: int):
    return scrape_multiple(filtered, max_workers=threads, max_return_chars=max_content_chars)


# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Robin: AI-Powered Dark Web OSINT Tool",
    page_icon="🕵️‍♂️",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .aStyle {
            font-size: 18px;
            font-weight: bold;
            padding: 5px;
            padding-left: 0px;
            text-align: left;
        }
    </style>""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Robin")
st.sidebar.text("AI-Powered Dark Web OSINT Tool")
st.sidebar.markdown("Made by [Apurv Singh Gautam](https://www.linkedin.com/in/apurvsinghgautam/)")
st.sidebar.subheader("Settings")


def _env_is_set(value) -> bool:
    return bool(value and str(value).strip() and "your_" not in str(value))


model_options = get_model_choices()
default_model_index = (
    next((idx for idx, name in enumerate(model_options) if name.lower() == "gpt4o"), 0)
    if model_options else 0
)

if not model_options:
    st.sidebar.error(
        "⛔ **No LLM models available.**\n\n"
        "No API keys or local providers are configured. "
        "Set at least one in your `.env` file and restart Robin.\n\n"
        "See **Provider Configuration** below for details."
    )
    st.stop()

model = st.sidebar.selectbox(
    "Select LLM Model", model_options, index=default_model_index, key="model_select"
)
if any(name not in {"gpt4o", "gpt-4.1", "claude-3-5-sonnet-latest", "llama3.1", "gemini-2.5-flash"} for name in model_options):
    st.sidebar.caption("Locally detected Ollama models are automatically added to this list.")

threads = st.sidebar.slider("Scraping Threads", 1, 16, 4, key="thread_slider")
max_results = st.sidebar.slider(
    "Max Results to Filter", 10, 100, 50, key="max_results_slider",
    help="Cap the number of raw search results passed to the LLM filter step.",
)
max_scrape = st.sidebar.slider(
    "Max Pages to Scrape", 3, 20, 10, key="max_scrape_slider",
    help="Cap the number of filtered results that get scraped for content.",
)
max_content_chars = st.sidebar.slider(
    "Content Size per Page", 2_000, 10_000, 2_000, step=1_000,
    key="max_content_chars_slider",
    help=(
        "Maximum characters kept from each scraped page before passing to the LLM. "
        "Higher values give more context but increase token usage and cost."
    ),
)

st.sidebar.divider()
st.sidebar.subheader("Provider Configuration")
_providers = [
    ("OpenAI",     OPENAI_API_KEY,     True),
    ("Anthropic",  ANTHROPIC_API_KEY,  True),
    ("Google",     GOOGLE_API_KEY,     True),
    ("OpenRouter", OPENROUTER_API_KEY, True),
    ("Ollama",     OLLAMA_BASE_URL,    False),
    ("llama.cpp",  LLAMA_CPP_BASE_URL, False),
]
for name, value, is_cloud in _providers:
    if _env_is_set(value):
        st.sidebar.markdown(f"&ensp;✅ **{name}** — configured")
    elif is_cloud:
        st.sidebar.markdown(f"&ensp;⚠️ **{name}** — API key not set")
    else:
        st.sidebar.markdown(f"&ensp;🔵 **{name}** — not configured *(optional)*")

with st.sidebar.expander("⚙️ Prompt Settings"):
    preset_options = {
        "🔍 Dark Web Threat Intel": "threat_intel",
        "🦠 Ransomware / Malware Focus": "ransomware_malware",
        "👤 Personal / Identity Investigation": "personal_identity",
        "🏢 Corporate Espionage / Data Leaks": "corporate_espionage",
    }
    preset_placeholders = {
        "threat_intel": "e.g. Pay extra attention to cryptocurrency wallet addresses and exchange names.",
        "ransomware_malware": "e.g. Highlight any references to double-extortion tactics or known ransomware-as-a-service affiliates.",
        "personal_identity": "e.g. Flag any passport or government ID numbers and note which country they appear to be from.",
        "corporate_espionage": "e.g. Prioritize any mentions of source code repositories, API keys, or internal Slack/email dumps.",
    }
    selected_preset_label = st.selectbox(
        "Research Domain", list(preset_options.keys()), key="preset_select"
    )
    selected_preset = preset_options[selected_preset_label]
    st.text_area(
        "System Prompt",
        value=PRESET_PROMPTS[selected_preset].strip(),
        height=200,
        disabled=True,
        key="system_prompt_display",
    )
    custom_instructions = st.text_area(
        "Custom Instructions (optional)",
        placeholder=preset_placeholders[selected_preset],
        height=100,
        key="custom_instructions",
    )

# --- Health Checks ---
st.sidebar.divider()
st.sidebar.subheader("Health Checks")

if st.sidebar.button("🔌 Check LLM Connection", use_container_width=True):
    with st.sidebar:
        with st.spinner(f"Testing {model}..."):
            result = check_llm_health(model)
        if result["status"] == "up":
            st.sidebar.success(f"✅ **{result['provider']}** — Connected ({result['latency_ms']}ms)")
        else:
            st.sidebar.error(f"❌ **{result['provider']}** — Failed\n\n{result['error']}")

if st.sidebar.button("🔍 Check Search Engines", use_container_width=True):
    with st.sidebar:
        with st.spinner("Checking Tor proxy..."):
            tor_result = check_tor_proxy()
        if tor_result["status"] == "down":
            st.sidebar.error(
                f"❌ **Tor Proxy** — Not reachable\n\n{tor_result['error']}\n\n"
                "Ensure Tor is running: `sudo systemctl start tor`"
            )
        else:
            st.sidebar.success(f"✅ **Tor Proxy** — Connected ({tor_result['latency_ms']}ms)")
            with st.spinner("Pinging 16 search engines via Tor..."):
                engine_results = check_search_engines()
            up_count = sum(1 for r in engine_results if r["status"] == "up")
            total = len(engine_results)
            if up_count == total:
                st.sidebar.success(f"✅ **All {total} engines reachable**")
            elif up_count > 0:
                st.sidebar.warning(f"⚠️ **{up_count}/{total} engines reachable**")
            else:
                st.sidebar.error(f"❌ **0/{total} engines reachable**")
            for r in engine_results:
                if r["status"] == "up":
                    st.sidebar.markdown(f"&ensp;🟢 **{r['name']}** — {r['latency_ms']}ms")
                else:
                    st.sidebar.markdown(f"&ensp;🔴 **{r['name']}** — {r['error']}")

# --- Past Investigations ---
st.sidebar.divider()
st.sidebar.subheader("📂 Past Investigations")
saved_investigations = load_investigations()
if saved_investigations:
    inv_labels = [
        f"{inv['_filename'].replace('investigation_','').replace('.json','')} — {inv['query'][:40]}"
        for inv in saved_investigations
    ]
    selected_inv_label = st.sidebar.selectbox(
        "Load investigation", ["(none)"] + inv_labels, key="inv_select"
    )
    if selected_inv_label != "(none)":
        selected_inv_idx = inv_labels.index(selected_inv_label)
        if st.sidebar.button("📂 Load", use_container_width=True, key="load_inv_btn"):
            st.session_state["loaded_investigation"] = saved_investigations[selected_inv_idx]
            st.rerun()
else:
    st.sidebar.caption("No saved investigations yet.")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

_, logo_col, _ = st.columns(3)
with logo_col:
    st.image(".github/assets/robin_logo.png", width=200)

with st.form("search_form", clear_on_submit=True):
    col_input, col_button = st.columns([10, 1])
    query = col_input.text_input(
        "Enter Dark Web Search Query",
        placeholder="Enter Dark Web Search Query",
        label_visibility="collapsed",
        key="query_input",
    )
    run_button = col_button.form_submit_button("Run")

# Display loaded investigation (if any)
if "loaded_investigation" in st.session_state and not run_button:
    inv = st.session_state["loaded_investigation"]
    st.info(f"📂 **{inv['query']}** — {inv['timestamp'][:16]}")
    with st.expander("📋 Notes", expanded=False):
        st.markdown(f"**Refined Query:** `{inv['refined_query']}`")
        st.markdown(f"**Model:** `{inv['model']}` &nbsp;&nbsp; **Domain:** {inv['preset']}")
        st.markdown(f"**Sources:** {len(inv['sources'])}")
    with st.expander(f"🔗 Sources ({len(inv['sources'])} results)", expanded=False):
        for i, item in enumerate(inv["sources"], 1):
            st.markdown(f"{i}. [{item.get('title', 'Untitled')}]({item.get('link', '')})")
    st.subheader(":red[🔎 Findings]", anchor=None, divider="gray")
    st.markdown(inv["summary"])
    if st.button("✖ Clear"):
        del st.session_state["loaded_investigation"]
        st.rerun()

# Placeholder slots rendered in order
status_slot = st.empty()
notes_placeholder = st.empty()
sources_placeholder = st.empty()
findings_placeholder = st.empty()


# ---------------------------------------------------------------------------
# Helper: render summary section (shared by full run and re-summarize)
# ---------------------------------------------------------------------------

def _render_findings(summary_text: str, filtered: list, query_text: str):
    """Render the findings expander and download link."""
    with findings_placeholder.container():
        st.subheader(":red[🔎 Findings]", anchor=None, divider="gray")
        st.markdown(summary_text)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"summary_{now}.md"
        b64 = base64.b64encode(summary_text.encode()).decode()
        href = (
            f'<div class="aStyle">📥 <a href="data:file/markdown;base64,{b64}" '
            f'download="{fname}">Download</a></div>'
        )
        st.markdown(href, unsafe_allow_html=True)


def _run_summary_stage(llm, query_text, scraped, preset, custom_instr, summary_slot):
    """Stream the summary into summary_slot and return the full text."""
    streamed = {"text": ""}

    def ui_emit(chunk: str):
        streamed["text"] += chunk
        summary_slot.markdown(streamed["text"])

    stream_handler = BufferedStreamingHandler(ui_callback=ui_emit)
    llm.callbacks = [stream_handler]
    generate_summary(llm, query_text, scraped, preset=preset, custom_instructions=custom_instr)
    return streamed["text"]


# ---------------------------------------------------------------------------
# Full pipeline (Run button)
# ---------------------------------------------------------------------------

if run_button and query:
    st.session_state.pop("loaded_investigation", None)
    for k in ["refined", "results", "filtered", "scraped", "streamed_summary"]:
        st.session_state.pop(k, None)

    # Stage 1 — Load LLM
    with status_slot.container():
        with st.spinner("🔄 Loading LLM..."):
            try:
                llm = get_llm(model)
            except Exception as e:
                _render_pipeline_error("load the selected LLM", e)

    # Stage 2 — Refine query
    with status_slot.container():
        with st.spinner("🔄 Refining query..."):
            try:
                st.session_state.refined = refine_query(llm, query)
            except Exception as e:
                _render_pipeline_error("refine the query", e)

    # Stage 3 — Search dark web
    with status_slot.container():
        with st.spinner("🔍 Searching dark web..."):
            st.session_state.results = cached_search_results(st.session_state.refined, threads)
    if len(st.session_state.results) > max_results:
        st.session_state.results = st.session_state.results[:max_results]

    # Stage 4 — Filter results (batched)
    with status_slot.container():
        num_batches = (len(st.session_state.results) + 24) // 25
        with st.spinner(f"🗂️ Filtering results ({num_batches} batch{'es' if num_batches > 1 else ''})..."):
            st.session_state.filtered = filter_results(
                llm, st.session_state.refined, st.session_state.results
            )
    if len(st.session_state.filtered) > max_scrape:
        st.session_state.filtered = st.session_state.filtered[:max_scrape]

    # Stage 5 — Scrape content
    with status_slot.container():
        with st.spinner("📜 Scraping content..."):
            st.session_state.scraped = cached_scrape_multiple(
                st.session_state.filtered, threads, max_content_chars
            )

    # Stage 6 — Streaming summary
    with findings_placeholder.container():
        st.subheader(":red[🔎 Findings]", anchor=None, divider="gray")
        summary_slot = st.empty()

    with status_slot.container():
        with st.spinner("✍️ Generating summary..."):
            st.session_state.streamed_summary = _run_summary_stage(
                llm, query, st.session_state.scraped,
                selected_preset, custom_instructions, summary_slot,
            )

    # Save investigation
    _fname = save_investigation(
        query=query,
        refined_query=st.session_state.refined,
        model=model,
        preset_label=selected_preset_label,
        sources=st.session_state.filtered,
        summary=st.session_state.streamed_summary,
    )

    # Render notes
    with notes_placeholder.container():
        with st.expander("📋 Notes", expanded=False):
            st.markdown(f"**Refined Query:** `{st.session_state.refined}`")
            st.markdown(f"**Model:** `{model}` &nbsp;&nbsp; **Domain:** {selected_preset_label}")
            st.markdown(
                f"**Results found:** {len(st.session_state.results)} &nbsp;&nbsp; "
                f"**Filtered to:** {len(st.session_state.filtered)} &nbsp;&nbsp; "
                f"**Scraped:** {len(st.session_state.scraped)} &nbsp;&nbsp; "
                f"**Content size:** {max_content_chars:,} chars/page"
            )

    # Render sources
    with sources_placeholder.container():
        with st.expander(f"🔗 Sources ({len(st.session_state.filtered)} results)", expanded=False):
            for i, item in enumerate(st.session_state.filtered, 1):
                st.markdown(f"{i}. [{item.get('title', 'Untitled')}]({item.get('link', '')})")

    # Re-render findings with download link
    _render_findings(st.session_state.streamed_summary, st.session_state.filtered, query)

    status_slot.success(f"✔️ Pipeline completed! Investigation saved as `{_fname}`")


# ---------------------------------------------------------------------------
# Re-summarize (skips search + scrape, reuses cached data)
# ---------------------------------------------------------------------------

# Show the Re-summarize button only when scraped data is already in session.
if st.session_state.get("scraped") and not run_button:
    st.divider()
    st.caption(
        "🔁 **Re-summarize** — change the model or prompt preset above, "
        "then click below to regenerate findings from the same scraped data."
    )
    resummarize_button = st.button(
        "🔁 Re-summarize with current settings",
        use_container_width=True,
        key="resummarize_btn",
    )

    if resummarize_button:
        with status_slot.container():
            with st.spinner("🔄 Loading LLM..."):
                try:
                    llm = get_llm(model)
                except Exception as e:
                    _render_pipeline_error("load the selected LLM", e)

        # Re-use the original query stored in session state.
        original_query = st.session_state.get("refined", "")

        with findings_placeholder.container():
            st.subheader(":red[🔎 Findings]", anchor=None, divider="gray")
            summary_slot = st.empty()

        with status_slot.container():
            with st.spinner("✍️ Re-generating summary..."):
                st.session_state.streamed_summary = _run_summary_stage(
                    llm, original_query, st.session_state.scraped,
                    selected_preset, custom_instructions, summary_slot,
                )

        # Save the re-generated version as a new investigation file.
        _fname = save_investigation(
            query=original_query,
            refined_query=st.session_state.get("refined", original_query),
            model=model,
            preset_label=selected_preset_label,
            sources=st.session_state.get("filtered", []),
            summary=st.session_state.streamed_summary,
        )

        _render_findings(
            st.session_state.streamed_summary,
            st.session_state.get("filtered", []),
            original_query,
        )
        status_slot.success(f"✔️ Re-summarized! Saved as `{_fname}`")