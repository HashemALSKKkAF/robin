import base64
import streamlit as st
from datetime import datetime

from scrape import scrape_multiple
from search import get_search_results
from llm_utils import BufferedStreamingHandler, get_model_choices
from llm import get_llm, refine_query, filter_results, generate_summary, PRESET_PROMPTS
from tor_utils import refresh_tor_circuit, get_tor_exit_ip
from scrape import get_tor_session
from export import generate_pdf
import investigations as inv_db
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


# ---------------------------------------------------------------------------
# Startup validation — warn about bad keys before anything else runs
# ---------------------------------------------------------------------------

def _validate_config() -> list:
    """
    Return a list of warning strings for obviously misconfigured values.
    Does not hard-stop the app — just surfaces issues early.
    """
    warnings = []

    def _check(name, value, prefix):
        if not value:
            return
        v = str(value).strip()
        if v.startswith("your_") or v.startswith(prefix + "_"):
            warnings.append(f"**{name}** looks like a placeholder value — double-check your `.env`.")
        if len(v) < 20 and "KEY" in name:
            warnings.append(f"**{name}** seems too short to be a valid API key.")
        if v != value:  # leading/trailing whitespace survived _clean_env
            warnings.append(f"**{name}** has surrounding whitespace — this will cause auth failures.")

    _check("OPENAI_API_KEY",     OPENAI_API_KEY,     "sk")
    _check("ANTHROPIC_API_KEY",  ANTHROPIC_API_KEY,  "sk-ant")
    _check("GOOGLE_API_KEY",     GOOGLE_API_KEY,     "AIza")
    _check("OPENROUTER_API_KEY", OPENROUTER_API_KEY, "sk-or")
    return warnings


# ---------------------------------------------------------------------------
# Error display helper
# ---------------------------------------------------------------------------

def _render_pipeline_error(stage: str, err: Exception) -> None:
    message = str(err).strip() or err.__class__.__name__
    lower_msg = message.lower()
    hints = [
        "- Confirm the relevant API key is set in your `.env` or shell before launching Streamlit.",
        "- Keys copied from dashboards often include hidden spaces; re-copy if authentication keeps failing.",
        "- Restart the app after updating environment variables so the new values are picked up.",
    ]
    if any(t in lower_msg for t in ("anthropic", "x-api-key", "invalid api key", "authentication")):
        hints.insert(0, "- Claude/Anthropic models require a valid `ANTHROPIC_API_KEY`.")
    elif "openrouter" in lower_msg or "user not found" in lower_msg or "code: 401" in lower_msg:
        hints.insert(0, "- OpenRouter 401 usually means an invalid/expired key or extra whitespace.")
    elif "openai" in lower_msg or "gpt" in lower_msg:
        hints.insert(0, "- OpenAI models require `OPENAI_API_KEY` with access to the chosen model.")
    elif "google" in lower_msg or "gemini" in lower_msg:
        hints.insert(0, "- Google Gemini needs `GOOGLE_API_KEY` or Application Default Credentials.")
    st.error("❌ Failed to {}.\n\nError: {}\n\n{}".format(stage, message, "\n".join(hints)))
    st.stop()


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
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Robin: AI-Powered Dark Web OSINT Tool",
    page_icon="🕵️‍♂️",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .aStyle {
        font-size: 18px; font-weight: bold;
        padding: 5px 0; text-align: left;
    }
</style>""", unsafe_allow_html=True)

# Run startup validation once per session
if "startup_warnings" not in st.session_state:
    st.session_state.startup_warnings = _validate_config()

if st.session_state.startup_warnings:
    with st.expander("⚠️ Configuration warnings — expand to review", expanded=False):
        for w in st.session_state.startup_warnings:
            st.warning(w)


# ---------------------------------------------------------------------------
# Sidebar — Settings
# ---------------------------------------------------------------------------

st.sidebar.title("Robin")
st.sidebar.text("AI-Powered Dark Web OSINT Tool")
st.sidebar.markdown("Made by [Apurv Singh Gautam](https://www.linkedin.com/in/apurvsinghgautam/)")
st.sidebar.subheader("Settings")


def _env_is_set(value) -> bool:
    return bool(value and str(value).strip() and "your_" not in str(value))


model_options = get_model_choices()
default_model_index = (
    next((i for i, n in enumerate(model_options) if n.lower() == "gpt4o"), 0)
    if model_options else 0
)

if not model_options:
    st.sidebar.error(
        "⛔ **No LLM models available.**\n\n"
        "No API keys or local providers are configured. "
        "Set at least one in your `.env` file and restart Robin."
    )
    st.stop()

model = st.sidebar.selectbox(
    "Select LLM Model", model_options, index=default_model_index, key="model_select"
)
if any(n not in {"gpt4o","gpt-4.1","claude-3-5-sonnet-latest","llama3.1","gemini-2.5-flash"} for n in model_options):
    st.sidebar.caption("Locally detected Ollama models are automatically added to this list.")

threads          = st.sidebar.slider("Scraping Threads", 1, 16, 4, key="thread_slider")
max_results      = st.sidebar.slider("Max Results to Filter", 10, 100, 50, key="max_results_slider",
                       help="Cap raw results sent to the LLM filter step.")
max_scrape       = st.sidebar.slider("Max Pages to Scrape", 3, 20, 10, key="max_scrape_slider",
                       help="Cap filtered results that get scraped.")
max_content_chars = st.sidebar.slider("Content Size per Page", 2_000, 10_000, 2_000, step=1_000,
                       key="max_content_chars_slider",
                       help="Max chars kept per scraped page. Higher = more context, more tokens.")

st.sidebar.divider()
st.sidebar.subheader("Provider Configuration")
for name, value, is_cloud in [
    ("OpenAI",     OPENAI_API_KEY,     True),
    ("Anthropic",  ANTHROPIC_API_KEY,  True),
    ("Google",     GOOGLE_API_KEY,     True),
    ("OpenRouter", OPENROUTER_API_KEY, True),
    ("Ollama",     OLLAMA_BASE_URL,    False),
    ("llama.cpp",  LLAMA_CPP_BASE_URL, False),
]:
    if _env_is_set(value):
        st.sidebar.markdown(f"&ensp;✅ **{name}** — configured")
    elif is_cloud:
        st.sidebar.markdown(f"&ensp;⚠️ **{name}** — API key not set")
    else:
        st.sidebar.markdown(f"&ensp;🔵 **{name}** — not configured *(optional)*")

with st.sidebar.expander("⚙️ Prompt Settings"):
    preset_options = {
        "🔍 Dark Web Threat Intel":          "threat_intel",
        "🦠 Ransomware / Malware Focus":      "ransomware_malware",
        "👤 Personal / Identity Investigation":"personal_identity",
        "🏢 Corporate Espionage / Data Leaks": "corporate_espionage",
    }
    preset_placeholders = {
        "threat_intel":       "e.g. Pay extra attention to cryptocurrency wallet addresses.",
        "ransomware_malware": "e.g. Highlight double-extortion tactics or RaaS affiliates.",
        "personal_identity":  "e.g. Flag passport numbers and note country of origin.",
        "corporate_espionage":"e.g. Prioritize source code repos, API keys, internal Slack dumps.",
    }
    selected_preset_label = st.selectbox("Research Domain", list(preset_options.keys()), key="preset_select")
    selected_preset       = preset_options[selected_preset_label]
    st.text_area("System Prompt", value=PRESET_PROMPTS[selected_preset].strip(),
                 height=200, disabled=True, key="system_prompt_display")
    custom_instructions = st.text_area("Custom Instructions (optional)",
                                        placeholder=preset_placeholders[selected_preset],
                                        height=100, key="custom_instructions")


# ---------------------------------------------------------------------------
# Sidebar — Health Checks
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("Health Checks")

if st.sidebar.button("🔌 Check LLM Connection", use_container_width=True):
    with st.sidebar, st.spinner(f"Testing {model}..."):
        result = check_llm_health(model)
    if result["status"] == "up":
        st.sidebar.success(f"✅ **{result['provider']}** — Connected ({result['latency_ms']}ms)")
    else:
        st.sidebar.error(f"❌ **{result['provider']}** — Failed\n\n{result['error']}")

if st.sidebar.button("🔍 Check Search Engines", use_container_width=True):
    with st.sidebar, st.spinner("Checking Tor proxy..."):
        tor_result = check_tor_proxy()
    if tor_result["status"] == "down":
        st.sidebar.error(
            f"❌ **Tor Proxy** — Not reachable\n\n{tor_result['error']}\n\n"
            "Ensure Tor is running: `sudo systemctl start tor`"
        )
    else:
        st.sidebar.success(f"✅ **Tor Proxy** — Connected ({tor_result['latency_ms']}ms)")
        with st.sidebar, st.spinner("Pinging 16 search engines via Tor..."):
            engine_results = check_search_engines()
        up_count = sum(1 for r in engine_results if r["status"] == "up")
        total    = len(engine_results)
        label    = f"✅ **All {total} engines reachable**" if up_count == total else \
                   f"⚠️ **{up_count}/{total} engines reachable**" if up_count else \
                   f"❌ **0/{total} engines reachable**"
        (st.sidebar.success if up_count == total else
         st.sidebar.warning if up_count else st.sidebar.error)(label)
        for r in engine_results:
            icon = "🟢" if r["status"] == "up" else "🔴"
            detail = f"{r['latency_ms']}ms" if r["status"] == "up" else r["error"]
            st.sidebar.markdown(f"&ensp;{icon} **{r['name']}** — {detail}")

if st.sidebar.button("🔄 Refresh Tor Circuit", use_container_width=True,
                     help="Send NEWNYM signal to rotate exit node. Requires ControlPort 9051 in torrc."):
    with st.sidebar, st.spinner("Requesting new Tor circuit..."):
        result = refresh_tor_circuit()
    if result["status"] == "ok":
        try:
            new_ip = get_tor_exit_ip(get_tor_session())
        except Exception:
            new_ip = None
        ip_line = f"\n\nNew exit IP: `{new_ip}`" if new_ip else ""
        st.sidebar.success(f"✅ {result['message']}{ip_line}")
    else:
        st.sidebar.error(
            f"❌ Circuit refresh failed\n\n{result['message']}\n\n"
            "Add `ControlPort 9051` to your torrc and restart Tor."
        )


# ---------------------------------------------------------------------------
# Sidebar — Past Investigations (SQLite-powered)
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("📂 Past Investigations")

# Filters
all_tags      = inv_db.get_all_tags()
filter_status = st.sidebar.selectbox("Filter by status", ["all"] + list(inv_db.VALID_STATUSES), key="filter_status")
filter_tag    = st.sidebar.selectbox("Filter by tag", ["all"] + all_tags, key="filter_tag") if all_tags else "all"

saved = inv_db.load_all(
    status_filter=None if filter_status == "all" else filter_status,
    tag_filter=None if filter_tag == "all" else filter_tag,
)

if saved:
    inv_labels = [
        f"{inv['timestamp'][:16]} — {inv['query'][:38]} [{inv['status']}]"
        for inv in saved
    ]
    selected_label = st.sidebar.selectbox("Load investigation", ["(none)"] + inv_labels, key="inv_select")
    if selected_label != "(none)":
        selected_inv = saved[inv_labels.index(selected_label)]
        if st.sidebar.button("📂 Load", use_container_width=True, key="load_inv_btn"):
            st.session_state["loaded_investigation"] = selected_inv
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
    query      = col_input.text_input("Enter Dark Web Search Query", placeholder="Enter Dark Web Search Query",
                                       label_visibility="collapsed", key="query_input")
    run_button = col_button.form_submit_button("Run")


# ---------------------------------------------------------------------------
# Loaded investigation view
# ---------------------------------------------------------------------------

if "loaded_investigation" in st.session_state and not run_button:
    inv = st.session_state["loaded_investigation"]
    inv_id = inv.get("id")

    st.info(f"📂 **{inv['query']}** — {inv['timestamp'][:16]}")

    # Meta + edit controls
    with st.expander("📋 Notes & Management", expanded=False):
        st.markdown(f"**Refined Query:** `{inv.get('refined_query','')}`")
        st.markdown(f"**Model:** `{inv.get('model','')}` &nbsp;&nbsp; **Domain:** {inv.get('preset','')}")
        st.markdown(f"**Sources:** {len(inv['sources'])}")

        col_s, col_t = st.columns(2)
        with col_s:
            new_status = st.selectbox("Status", inv_db.VALID_STATUSES,
                                       index=list(inv_db.VALID_STATUSES).index(inv.get("status","active")),
                                       key="edit_status")
        with col_t:
            new_tags = st.text_input("Tags (comma-separated)", value=inv.get("tags",""), key="edit_tags")

        if st.button("💾 Save changes", key="save_meta_btn"):
            inv_db.update_status(inv_id, new_status)
            inv_db.update_tags(inv_id, new_tags)
            st.success("Status and tags updated.")
            st.session_state["loaded_investigation"] = inv_db.load_one(inv_id)
            st.rerun()

        if st.button("🗑️ Delete investigation", key="delete_inv_btn", type="secondary"):
            inv_db.delete_investigation(inv_id)
            del st.session_state["loaded_investigation"]
            st.rerun()

    with st.expander(f"🔗 Sources ({len(inv['sources'])} results)", expanded=False):
        for i, item in enumerate(inv["sources"], 1):
            st.markdown(f"{i}. [{item.get('title','Untitled')}]({item.get('link','')})")

    st.subheader(":red[🔎 Findings]", anchor=None, divider="gray")
    st.markdown(inv["summary"])

    # PDF export for loaded investigation
    col_md, col_pdf = st.columns(2)
    with col_md:
        b64 = base64.b64encode(inv["summary"].encode()).decode()
        st.markdown(
            f'<div class="aStyle">📥 <a href="data:file/markdown;base64,{b64}" '
            f'download="summary_{inv["timestamp"][:10]}.md">Download Markdown</a></div>',
            unsafe_allow_html=True,
        )
    with col_pdf:
        pdf_bytes = generate_pdf(inv)
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"robin_investigation_{inv['timestamp'][:10]}.pdf",
            mime="application/pdf",
            key="pdf_dl_loaded",
        )

    if st.button("✖ Clear"):
        del st.session_state["loaded_investigation"]
        st.rerun()


# ---------------------------------------------------------------------------
# Pipeline placeholders
# ---------------------------------------------------------------------------

status_slot         = st.empty()
notes_placeholder   = st.empty()
sources_placeholder = st.empty()
findings_placeholder = st.empty()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _render_findings(summary_text: str, inv_dict: dict):
    """Render findings section with both Markdown and PDF download buttons."""
    with findings_placeholder.container():
        st.subheader(":red[🔎 Findings]", anchor=None, divider="gray")
        st.markdown(summary_text)

        col_md, col_pdf = st.columns(2)
        now = datetime.now().strftime("%Y-%m-%d")
        with col_md:
            b64 = base64.b64encode(summary_text.encode()).decode()
            st.markdown(
                f'<div class="aStyle">📥 <a href="data:file/markdown;base64,{b64}" '
                f'download="summary_{now}.md">Download Markdown</a></div>',
                unsafe_allow_html=True,
            )
        with col_pdf:
            pdf_bytes = generate_pdf(inv_dict)
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"robin_investigation_{now}.pdf",
                mime="application/pdf",
                key=f"pdf_dl_{now}_{hash(summary_text) % 99999}",
            )


def _run_summary_stage(llm, query_text, scraped, preset, custom_instr, summary_slot):
    """Stream the LLM summary into summary_slot and return the full accumulated text."""
    streamed = {"text": ""}

    def ui_emit(chunk: str):
        streamed["text"] += chunk
        summary_slot.markdown(streamed["text"])

    stream_handler = BufferedStreamingHandler(ui_callback=ui_emit)
    llm.callbacks  = [stream_handler]
    generate_summary(llm, query_text, scraped, preset=preset, custom_instructions=custom_instr)
    return streamed["text"]


# ---------------------------------------------------------------------------
# Full pipeline — uses st.status for expandable live stage feedback
# ---------------------------------------------------------------------------

if run_button and query:
    st.session_state.pop("loaded_investigation", None)
    for k in ["refined", "results", "filtered", "scraped", "streamed_summary", "last_inv_id"]:
        st.session_state.pop(k, None)

    with st.status("🔄 Running investigation pipeline…", expanded=True) as pipeline_status:

        # Stage 1 — Load LLM
        st.write("🔄 Loading LLM…")
        try:
            llm = get_llm(model)
        except Exception as e:
            pipeline_status.update(label="❌ Pipeline failed", state="error", expanded=True)
            _render_pipeline_error("load the selected LLM", e)

        # Stage 2 — Refine query
        st.write("✏️ Refining query…")
        try:
            st.session_state.refined = refine_query(llm, query)
            st.write(f"&nbsp;&nbsp;&nbsp;→ `{st.session_state.refined}`")
        except Exception as e:
            pipeline_status.update(label="❌ Pipeline failed", state="error", expanded=True)
            _render_pipeline_error("refine the query", e)

        # Stage 3 — Search
        st.write("🔍 Searching dark web (results pre-scored by keyword relevance)…")
        st.session_state.results = cached_search_results(st.session_state.refined, threads)
        if len(st.session_state.results) > max_results:
            st.session_state.results = st.session_state.results[:max_results]
        st.write(f"&nbsp;&nbsp;&nbsp;→ {len(st.session_state.results)} unique results")

        # Stage 4 — Filter (batched)
        num_batches = (len(st.session_state.results) + 24) // 25
        st.write(f"🗂️ Filtering results ({num_batches} batch{'es' if num_batches > 1 else ''})…")
        st.session_state.filtered = filter_results(
            llm, st.session_state.refined, st.session_state.results
        )
        if len(st.session_state.filtered) > max_scrape:
            st.session_state.filtered = st.session_state.filtered[:max_scrape]
        st.write(f"&nbsp;&nbsp;&nbsp;→ {len(st.session_state.filtered)} selected")

        # Stage 5 — Scrape
        st.write("📜 Scraping content…")
        st.session_state.scraped = cached_scrape_multiple(
            st.session_state.filtered, threads, max_content_chars
        )
        st.write(f"&nbsp;&nbsp;&nbsp;→ {len(st.session_state.scraped)} pages scraped")

        # Stage 6 — Summary (streamed into findings area while status is still open)
        st.write("✍️ Generating summary…")
        with findings_placeholder.container():
            st.subheader(":red[🔎 Findings]", anchor=None, divider="gray")
            summary_slot = st.empty()

        st.session_state.streamed_summary = _run_summary_stage(
            llm, query, st.session_state.scraped,
            selected_preset, custom_instructions, summary_slot,
        )

        pipeline_status.update(label="✅ Pipeline complete", state="complete", expanded=False)

    # Save to SQLite
    inv_id = inv_db.save_investigation(
        query=query,
        refined_query=st.session_state.refined,
        model=model,
        preset_label=selected_preset_label,
        sources=st.session_state.filtered,
        summary=st.session_state.streamed_summary,
    )
    st.session_state["last_inv_id"] = inv_id

    # Notes
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

            # Inline tag/status editing immediately after a run
            st.markdown("---")
            col_s, col_t = st.columns(2)
            with col_s:
                quick_status = st.selectbox("Set status", inv_db.VALID_STATUSES, key="quick_status")
            with col_t:
                quick_tags = st.text_input("Add tags (comma-separated)", key="quick_tags")
            if st.button("💾 Save status & tags", key="quick_save_btn"):
                inv_db.update_status(inv_id, quick_status)
                inv_db.update_tags(inv_id, quick_tags)
                st.success("Saved.")

    # Sources
    with sources_placeholder.container():
        with st.expander(f"🔗 Sources ({len(st.session_state.filtered)} results)", expanded=False):
            for i, item in enumerate(st.session_state.filtered, 1):
                st.markdown(f"{i}. [{item.get('title','Untitled')}]({item.get('link','')})")

    # Findings + download buttons
    inv_dict_for_pdf = {
        "query":         query,
        "refined_query": st.session_state.refined,
        "model":         model,
        "preset":        selected_preset_label,
        "status":        "active",
        "tags":          "",
        "timestamp":     datetime.now().isoformat(),
        "sources":       st.session_state.filtered,
        "summary":       st.session_state.streamed_summary,
    }
    _render_findings(st.session_state.streamed_summary, inv_dict_for_pdf)


# ---------------------------------------------------------------------------
# Re-summarize — reuses cached scraped data, skips search + scrape
# ---------------------------------------------------------------------------

if st.session_state.get("scraped") and not run_button:
    st.divider()
    st.caption(
        "🔁 **Re-summarize** — change the model or preset above, "
        "then click below to regenerate findings from the same scraped data."
    )
    if st.button("🔁 Re-summarize with current settings", use_container_width=True, key="resummarize_btn"):

        with st.status("✍️ Re-generating summary…", expanded=True) as rs_status:
            st.write("🔄 Loading LLM…")
            try:
                llm = get_llm(model)
            except Exception as e:
                rs_status.update(label="❌ Failed", state="error")
                _render_pipeline_error("load the selected LLM", e)

            original_query = st.session_state.get("refined", "")
            st.write("✍️ Streaming new summary…")

            with findings_placeholder.container():
                st.subheader(":red[🔎 Findings]", anchor=None, divider="gray")
                summary_slot = st.empty()

            st.session_state.streamed_summary = _run_summary_stage(
                llm, original_query, st.session_state.scraped,
                selected_preset, custom_instructions, summary_slot,
            )
            rs_status.update(label="✅ Re-summarization complete", state="complete", expanded=False)

        # Save re-generated version
        new_inv_id = inv_db.save_investigation(
            query=original_query,
            refined_query=st.session_state.get("refined", original_query),
            model=model,
            preset_label=selected_preset_label,
            sources=st.session_state.get("filtered", []),
            summary=st.session_state.streamed_summary,
        )
        st.session_state["last_inv_id"] = new_inv_id

        inv_dict_for_pdf = {
            "query":         original_query,
            "refined_query": st.session_state.get("refined", original_query),
            "model":         model,
            "preset":        selected_preset_label,
            "status":        "active",
            "tags":          "",
            "timestamp":     datetime.now().isoformat(),
            "sources":       st.session_state.get("filtered", []),
            "summary":       st.session_state.streamed_summary,
        }
        _render_findings(st.session_state.streamed_summary, inv_dict_for_pdf)