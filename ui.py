import base64
import hashlib
import os
import platform
import subprocess
import tempfile
import streamlit as st
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from scrape import scrape_multiple
from search import get_search_results
from llm_utils import BufferedStreamingHandler, get_model_choices
from llm import get_llm, refine_query, filter_results, generate_summary, PRESET_PROMPTS
from export import generate_pdf
import investigations as inv_db
import seeds as seed_db
import presets as preset_db
from crawler import crawl_sources, crawl_url, probe_tier
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
# Startup validation
# ---------------------------------------------------------------------------

def _validate_config() -> list:
    warnings = []
    def _check(name, value, prefix):
        if not value:
            return
        v = str(value).strip()
        if v.startswith("your_") or v.startswith(prefix + "_"):
            warnings.append(f"**{name}** looks like a placeholder — double-check your `.env`.")
        if len(v) < 20 and "KEY" in name:
            warnings.append(f"**{name}** seems too short to be a valid API key.")
    _check("OPENAI_API_KEY",     OPENAI_API_KEY,     "sk")
    _check("ANTHROPIC_API_KEY",  ANTHROPIC_API_KEY,  "sk-ant")
    _check("GOOGLE_API_KEY",     GOOGLE_API_KEY,     "AIza")
    _check("OPENROUTER_API_KEY", OPENROUTER_API_KEY, "sk-or")
    return warnings


def _open_path_in_system_app(path: str) -> tuple[bool, str]:
    """
    Hand `path` to the OS so the user's default app (text editor for .txt,
    browser for .html, etc.) opens it. Server-side: only useful when
    Streamlit runs on the same machine the user is sitting at.

    Returns (ok, message).
    """
    try:
        system = platform.system()
        if system == "Darwin":
            # `-t` forces a text editor (better default for inspecting content).
            subprocess.Popen(["open", "-t", path])
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            # Linux / BSD — xdg-open picks the user's registered handler.
            subprocess.Popen(["xdg-open", path])
        return True, f"Launched system handler for `{path}`."
    except FileNotFoundError as exc:
        return False, f"Couldn't find a system-open command: {exc}"
    except Exception as exc:
        return False, f"Could not open `{path}`: {exc}"


def _crawled_html_path(url: str) -> Path | None:
    """Return the path to the saved rendered.html for `url`, or None if absent."""
    h = hashlib.sha256(url.encode()).hexdigest()
    candidate = Path("investigations") / "crawled" / h / "rendered.html"
    return candidate if candidate.exists() else None


def _render_pipeline_error(stage: str, err: Exception) -> None:
    message = str(err).strip() or err.__class__.__name__
    lower_msg = message.lower()
    hints = [
        "- Confirm the relevant API key is set in your `.env` before launching.",
        "- Keys copied from dashboards often include hidden spaces — re-copy if auth keeps failing.",
        "- Restart the app after updating environment variables.",
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
    # Percent-encode the query so special characters (& ? = # /) don't break
    # the engine URL templates that interpolate it as `?q={query}`.
    return get_search_results(quote(refined_query, safe=""), max_workers=threads)


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
    .aStyle { font-size:18px; font-weight:bold; padding:5px 0; text-align:left; }
    .deep-crawl-box {
        border: 1px solid #444; border-radius: 8px;
        padding: 12px 16px; margin-top: 10px; background: #111;
    }
</style>""", unsafe_allow_html=True)

if "startup_warnings" not in st.session_state:
    st.session_state.startup_warnings = _validate_config()
if st.session_state.startup_warnings:
    with st.expander("⚠️ Configuration warnings", expanded=False):
        for w in st.session_state.startup_warnings:
            st.warning(w)


# ---------------------------------------------------------------------------
# Sidebar — Settings
# ---------------------------------------------------------------------------

st.sidebar.title("Robin")
st.sidebar.text("AI-Powered Dark Web OSINT Tool")
st.sidebar.markdown("Made by [Apurv Singh Gautam](https://www.linkedin.com/in/apurvsinghgautam/)")
st.sidebar.subheader("Settings")


def _env_is_set(v) -> bool:
    return bool(v and str(v).strip() and "your_" not in str(v))


model_options = get_model_choices()
if not model_options:
    st.sidebar.error("⛔ No LLM models available. Set at least one API key in `.env` and restart.")
    st.stop()

# Preferred default models, in priority order. The first one that's actually
# present in `model_options` (i.e. has its API key configured) wins.
_PREFERRED_DEFAULTS = (
    "claude-sonnet-4-5",
    "gpt-5.1",
    "gpt-4.1",
    "gemini-2.5-pro",
    "claude-sonnet-4.5-openrouter",
)
default_model_index = next(
    (model_options.index(m) for m in _PREFERRED_DEFAULTS if m in model_options),
    0,
)
model = st.sidebar.selectbox("Select LLM Model", model_options, index=default_model_index, key="model_select")

threads           = st.sidebar.slider("Scraping Threads", 1, 16, 4, key="thread_slider")
max_results       = st.sidebar.slider("Max Results to Filter", 10, 100, 50, key="max_results_slider",
                        help="Cap raw results sent to the LLM filter step.")
max_scrape        = st.sidebar.slider("Max Pages to Scrape", 3, 20, 10, key="max_scrape_slider",
                        help="Cap filtered results that get scraped.")
max_content_chars = st.sidebar.slider("Content Size per Page", 2_000, 10_000, 2_000, step=1_000,
                        key="max_content_chars_slider",
                        help="Max chars kept per scraped page.")

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

# Built-in domains. These four MUST keep their existing keys + behavior, so
# they're hard-coded here exactly as before.
_BUILTIN_PRESETS = {
    "🔍 Dark Web Threat Intel":           "threat_intel",
    "🦠 Ransomware / Malware Focus":       "ransomware_malware",
    "👤 Personal / Identity Investigation": "personal_identity",
    "🏢 Corporate Espionage / Data Leaks":  "corporate_espionage",
}
_BUILTIN_PLACEHOLDERS = {
    "threat_intel":       "e.g. Pay extra attention to cryptocurrency wallet addresses.",
    "ransomware_malware": "e.g. Highlight double-extortion tactics or RaaS affiliates.",
    "personal_identity":  "e.g. Flag passport numbers and note country of origin.",
    "corporate_espionage":"e.g. Prioritize source code repos, API keys, Slack dumps.",
}

with st.sidebar.expander("⚙️ Prompt Settings"):
    custom_presets = preset_db.list_presets()

    # Build the dropdown: built-ins first (preserves their existing labels),
    # then any user-defined domains prefixed with ✨.
    label_to_key: dict[str, str] = dict(_BUILTIN_PRESETS)
    for cp in custom_presets:
        label_to_key[f"✨ {cp['name']}"] = cp["key"]

    selected_preset_label = st.selectbox(
        "Research Domain", list(label_to_key.keys()), key="preset_select",
    )
    selected_preset = label_to_key[selected_preset_label]
    is_custom = preset_db.is_custom_key(selected_preset)

    if is_custom:
        cp = preset_db.get_preset_by_key(selected_preset)
        if cp is None:
            # Stale selectbox state — shouldn't happen, but recover gracefully.
            st.warning("This custom domain no longer exists. Pick another.")
            selected_preset = "threat_intel"
            cp = None
            current_system_prompt = PRESET_PROMPTS["threat_intel"]
        else:
            current_system_prompt = cp["system_prompt"]
            st.caption(
                f"Created {cp['created_at'][:10]}"
                + (f" · updated {cp['updated_at'][:10]}"
                   if cp['updated_at'] != cp['created_at'] else "")
            )

        # Editable system prompt for custom domains.
        edited = st.text_area(
            "System Prompt (editable)",
            value=current_system_prompt,
            height=240,
            key=f"system_prompt_edit_{cp['id'] if cp else 'none'}",
        )
        edited_desc = st.text_input(
            "Description (optional)",
            value=cp.get("description", "") if cp else "",
            key=f"desc_edit_{cp['id'] if cp else 'none'}",
        )

        col_save, col_del = st.columns(2)
        with col_save:
            if cp and st.button("💾 Save", key=f"save_preset_{cp['id']}",
                                use_container_width=True):
                try:
                    preset_db.update_preset(
                        cp["id"],
                        system_prompt=edited,
                        description=edited_desc,
                    )
                    st.success("Saved.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        with col_del:
            if cp and st.button("🗑️ Delete", key=f"del_preset_{cp['id']}",
                                use_container_width=True, type="secondary"):
                preset_db.delete_preset(cp["id"])
                st.success(f"Deleted '{cp['name']}'.")
                st.rerun()

        custom_placeholder = (
            "e.g. Cross-reference any leaked email addresses against known "
            "breach corpora and flag overlapping infrastructure."
        )
    else:
        # Built-in domain — keep the legacy read-only display verbatim.
        current_system_prompt = PRESET_PROMPTS[selected_preset]
        st.text_area(
            "System Prompt", value=current_system_prompt.strip(),
            height=200, disabled=True, key="system_prompt_display",
        )
        custom_placeholder = _BUILTIN_PLACEHOLDERS[selected_preset]

    custom_instructions = st.text_area(
        "Custom Instructions (optional)",
        placeholder=custom_placeholder,
        height=100,
        key="custom_instructions",
    )

    # ── Create new custom domain ─────────────────────────────────────────
    st.divider()
    st.caption("➕ Create a new custom research domain")
    with st.form("new_preset_form", clear_on_submit=True):
        new_preset_name = st.text_input(
            "Domain name",
            placeholder="e.g. Crypto Tracing",
        )
        new_preset_desc = st.text_input(
            "Description (optional)",
            placeholder="One-line summary of what this domain focuses on",
        )
        new_preset_prompt = st.text_area(
            "System prompt",
            height=200,
            placeholder=(
                "Describe the analyst persona, the analysis rules, and the "
                "output format. {query} will be substituted with the user's "
                "query at runtime."
            ),
        )
        if st.form_submit_button("➕ Create domain"):
            try:
                preset_db.create_preset(
                    new_preset_name, new_preset_prompt, new_preset_desc,
                )
                st.success(f"Created '{new_preset_name.strip()}'.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

# Resolve the active system prompt once, after the expander block, so the
# rest of the script can pass it straight into generate_summary regardless
# of whether the user picked a built-in or custom domain.
_active_custom = (
    preset_db.get_preset_by_key(selected_preset)
    if preset_db.is_custom_key(selected_preset) else None
)
selected_system_prompt = (
    _active_custom["system_prompt"] if _active_custom
    else PRESET_PROMPTS.get(selected_preset, PRESET_PROMPTS["threat_intel"])
)


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
        st.sidebar.error(f"❌ **Tor Proxy** — Not reachable\n\n{tor_result['error']}")
    else:
        st.sidebar.success(f"✅ **Tor Proxy** — Connected ({tor_result['latency_ms']}ms)")
        with st.sidebar, st.spinner("Pinging search engines..."):
            engine_results = check_search_engines()
        up_count = sum(1 for r in engine_results if r["status"] == "up")
        total    = len(engine_results)
        lbl = (f"✅ **All {total} engines reachable**" if up_count == total else
               f"⚠️ **{up_count}/{total} engines reachable**" if up_count else
               f"❌ **0/{total} engines reachable**")
        (st.sidebar.success if up_count == total else
         st.sidebar.warning if up_count else st.sidebar.error)(lbl)
        for r in engine_results:
            icon   = "🟢" if r["status"] == "up" else "🔴"
            detail = f"{r['latency_ms']}ms" if r["status"] == "up" else r["error"]
            st.sidebar.markdown(f"&ensp;{icon} **{r['name']}** — {detail}")

# ---------------------------------------------------------------------------
# Sidebar — Seed Manager
# ---------------------------------------------------------------------------

st.sidebar.divider()
with st.sidebar.expander("🌱 Seed Manager", expanded=False):
    st.caption("Add .onion URLs and inspect previously deep-crawled content.")

    with st.form("add_seed_form", clear_on_submit=True):
        new_url  = st.text_input("URL", placeholder="http://example.onion")
        new_name = st.text_input("Label (optional)")
        if st.form_submit_button("➕ Add Seed"):
            if new_url.strip():
                try:
                    seed_db.add_seed(new_url.strip(), new_name.strip())
                    st.success("Seed added.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a URL.")

    all_seeds = seed_db.get_all_seeds()
    if all_seeds:
        # Filter dropdown — same UX as Past Investigations.
        seed_filter = st.selectbox(
            "Filter seeds",
            ["all", "crawled", "uncrawled", "with content"],
            key="seed_filter",
        )
        if seed_filter == "crawled":
            visible = [s for s in all_seeds if s["crawled"]]
        elif seed_filter == "uncrawled":
            visible = [s for s in all_seeds if not s["crawled"]]
        elif seed_filter == "with content":
            visible = [s for s in all_seeds if (s.get("content") or "")]
        else:
            visible = all_seeds

        st.caption(f"**{len(visible)} / {len(all_seeds)} seeds**")

        if visible:
            def _seed_label(s):
                icon  = "✅" if s["crawled"] else "⏳"
                doc   = "📄" if (s.get("content") or "") else "—"
                short = s["url"][:34] + ("…" if len(s["url"]) > 34 else "")
                return f"{icon}{doc} {short}"

            seed_labels = [_seed_label(s) for s in visible]
            # Keying by filter ensures the picker resets when filter changes,
            # so it can't reference an item that's been filtered out.
            seed_key = f"seed_select_{seed_filter}"
            chosen = st.selectbox(
                "Open seed", ["(none)"] + seed_labels, key=seed_key,
            )
            if chosen != "(none)":
                try:
                    s = visible[seed_labels.index(chosen)]
                except ValueError:
                    s = None
                if s:
                    status_word = "crawled" if s["crawled"] else "pending"
                    code = s.get("status_code")
                    code_part = f" (HTTP {code})" if code else ""
                    crawled_at = s.get("crawled_at")
                    crawled_line = (
                        f"  \n**Last crawled:** {crawled_at[:16]}"
                        if crawled_at else ""
                    )
                    st.markdown(f"**Label:** {s.get('name') or '—'}")
                    st.markdown(
                        f"**URL:** `{s['url']}`  \n"
                        f"**Status:** {status_word}{code_part}  \n"
                        f"**Added:** {s['added_at'][:16]}"
                        f"{crawled_line}"
                    )

                    # ── Open in browser tab ────────────────────────────────
                    # Plain HTML link with target="_blank" — clicked in the
                    # user's browser, so it opens in whichever browser they
                    # have (Tor Browser handles .onion; otherwise it'll fail
                    # in a normal browser, which is on them, not us).
                    st.markdown(
                        f'🌐 <a href="{s["url"]}" target="_blank" '
                        f'rel="noopener noreferrer">Open URL in new browser tab ↗</a>',
                        unsafe_allow_html=True,
                    )

                    content = s.get("content") or ""
                    if content:
                        st.caption(f"📄 {len(content):,} chars saved")
                        with st.expander("View extracted content", expanded=False):
                            st.text(content[:5000] + ("…" if len(content) > 5000 else ""))
                    else:
                        st.caption("No content saved yet — run Deep Crawl to populate.")

                    # ── Open extracted content in system editor ────────────
                    # Writes the saved text to a temp .txt file then asks
                    # the OS to launch the user's default text editor.
                    # Only meaningful when streamlit runs on the user's own
                    # machine (typical for local OSINT use).
                    if content:
                        if st.button(
                            "📝 Open content in system editor",
                            key=f"editor_open_{s['id']}",
                            help="Writes the extracted text to a temp .txt and "
                                 "calls the OS handler (xdg-open / open / startfile).",
                        ):
                            try:
                                fd, tmp = tempfile.mkstemp(
                                    prefix=f"robin_seed_{s['id']}_",
                                    suffix=".txt",
                                )
                                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                                    fh.write(f"# {s['url']}\n")
                                    fh.write(f"# Crawled: {s.get('crawled_at') or '—'}\n")
                                    fh.write(f"# {len(content):,} chars\n\n")
                                    fh.write(content)
                                ok, msg = _open_path_in_system_app(tmp)
                                if ok:
                                    st.success(f"📝 {msg}")
                                else:
                                    st.error(f"❌ {msg}\n\nFile saved at `{tmp}`.")
                            except Exception as exc:
                                st.error(f"❌ Could not prepare temp file: {exc}")

                    # If the saved rendered.html still exists, offer to open
                    # that in the user's default app too (typically a browser
                    # — handy for inspecting the original markup).
                    html_path = _crawled_html_path(s["url"])
                    if html_path is not None:
                        if st.button(
                            "🌐 Open saved HTML in default app",
                            key=f"html_open_{s['id']}",
                            help=f"Asks the OS to open {html_path}",
                        ):
                            ok, msg = _open_path_in_system_app(str(html_path))
                            (st.success if ok else st.error)(msg)

                    if st.button("🗑️ Delete seed", key=f"del_seed_{s['id']}"):
                        seed_db.delete_seed(s["id"])
                        st.rerun()
        else:
            st.caption("No seeds match this filter.")
    else:
        st.caption("No seeds yet.")


# ---------------------------------------------------------------------------
# Sidebar — Past Investigations
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("📂 Past Investigations")
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
    # Including the active filters in the widget key forces Streamlit to
    # reset the picker when the filter set changes — otherwise a previously
    # selected label can persist into a filter set that no longer contains
    # it, which causes the load selectbox to render blank/freeze on scroll.
    inv_select_key = f"inv_select_{filter_status}_{filter_tag}"
    selected_label = st.sidebar.selectbox(
        "Load investigation", ["(none)"] + inv_labels, key=inv_select_key,
    )
    if selected_label != "(none)":
        try:
            selected_inv = saved[inv_labels.index(selected_label)]
        except ValueError:
            selected_inv = None
        if selected_inv and st.sidebar.button(
            "📂 Load", use_container_width=True, key="load_inv_btn",
        ):
            st.session_state["loaded_investigation"] = selected_inv
            # Clear any leftover fresh-run state so the loaded view
            # doesn't accidentally re-summarize stale pipeline data.
            for k in ("refined", "results", "filtered", "scraped",
                      "streamed_summary", "last_inv_id"):
                st.session_state.pop(k, None)
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
    query      = col_input.text_input("Enter Dark Web Search Query",
                                       placeholder="Enter Dark Web Search Query",
                                       label_visibility="collapsed", key="query_input")
    run_button = col_button.form_submit_button("Run")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _render_findings(summary_text: str, inv_dict: dict, target=None):
    """
    Render findings with Markdown + PDF download buttons.

    `target` is the placeholder/container to render into. If omitted, the
    module-level `findings_placeholder` is used (which only exists once
    the script reaches the pipeline-placeholder block, so callers in the
    loaded-investigation view must always pass an explicit target).
    """
    container = target.container() if target is not None else findings_placeholder.container()
    with container:
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
                "📄 Download PDF Report", data=pdf_bytes,
                file_name=f"robin_investigation_{now}.pdf",
                mime="application/pdf",
                key=f"pdf_dl_{now}_{abs(hash(summary_text)) % 99999}",
            )


def _run_summary_stage(
    llm, query_text, scraped, preset, custom_instr, summary_slot,
    system_prompt_override: str | None = None,
):
    """
    Stream the LLM summary into summary_slot, return full accumulated text.

    `system_prompt_override`, if given, is used verbatim as the system prompt
    — that's how user-defined custom research domains flow through.
    Built-in domains continue to resolve via `preset` against PRESET_PROMPTS.
    """
    streamed = {"text": ""}
    def ui_emit(chunk: str):
        streamed["text"] += chunk
        summary_slot.markdown(streamed["text"])
    stream_handler = BufferedStreamingHandler(ui_callback=ui_emit)
    llm.callbacks  = [stream_handler]
    generate_summary(
        llm, query_text, scraped,
        preset=preset, custom_instructions=custom_instr,
        system_prompt_override=system_prompt_override,
    )
    return streamed["text"]


def _deep_crawl_sources_section(
    sources: list,
    scraped_key: str,
    query_text: str,
    section_label: str = "sources",
):
    """
    Render the deep crawl UI block below a sources list.
    - Shows the crawl tier detected (Selenium vs requests).
    - Lets the user crawl individual sources or all at once.
    - On completion, re-runs the LLM summary with deep-crawled content
      merged into the existing scraped data and re-renders findings.

    Args:
        sources:      List of {"link":..., "title":...} dicts.
        scraped_key:  session_state key holding the existing scraped dict
                      (so we can merge deep content into it).
        query_text:   Query used for the summary stage.
        section_label: Display label for messages.
    """
    tier = probe_tier()
    tier_label = (
        "🦊 **Tier 1 — Tor Browser** (full JS rendering)"
        if tier == "selenium"
        else "🌐 **Tier 2 — requests + SOCKS** (lightweight fallback)"
    )

    st.markdown(f"**Deep Crawl available** &nbsp;|&nbsp; {tier_label}")
    if tier == "requests":
        st.caption(
            "Tor Browser + geckodriver not detected. Using fast requests-based crawl. "
            "Set `TORBROWSER_BINARY` env var and install geckodriver to enable Tier 1."
        )

    # Per-source crawl buttons
    with st.expander(f"🔗 Crawl individual {section_label}", expanded=False):
        for i, src in enumerate(sources):
            url   = src.get("link", "")
            title = src.get("title", "Untitled")
            if not url:
                continue
            col_lbl, col_btn = st.columns([8, 2])
            with col_lbl:
                st.markdown(f"**{i+1}.** {title[:60]}{'…' if len(title)>60 else ''}")
                st.caption(url[:70] + ("…" if len(url) > 70 else ""))
            with col_btn:
                if st.button("🕷️ Crawl", key=f"crawl_single_{scraped_key}_{i}"):
                    with st.spinner(f"Deep crawling {url[:40]}…"):
                        result = crawl_url(url, title_hint=title, tier=tier)
                    if result["success"]:
                        existing = st.session_state.get(scraped_key, {})
                        existing[url] = result["text"]
                        st.session_state[scraped_key] = existing
                        # Persist the crawl into the seeds DB so it survives reloads.
                        saved_to_db = False
                        try:
                            seed_db.add_seed(url, title)
                            sid = seed_db.get_seed_by_url(url)
                            if sid:
                                seed_db.mark_crawled(
                                    sid["id"], status_code=200, content=result["text"],
                                )
                                saved_to_db = True
                        except Exception:
                            pass
                        save_msg = (
                            "Saved to **Seed Manager** (sidebar) and HTML archived to "
                            "`investigations/crawled/<hash>/`."
                            if saved_to_db
                            else "HTML archived to `investigations/crawled/<hash>/` "
                                 "(could not save to Seed Manager)."
                        )
                        st.success(
                            f"✅ Crawled `{url[:40]}` — {len(result['text']):,} chars via "
                            f"{result['tier']}.\n\n{save_msg}\n\n"
                            "Click **Re-summarize** below to update findings."
                        )
                    else:
                        st.error(f"❌ Failed: {result['error']}")

    # Crawl all sources at once
    if st.button(
        f"🕷️ Deep Crawl all {len(sources)} {section_label}",
        key=f"crawl_all_{scraped_key}",
        use_container_width=True,
    ):
        progress_bar  = st.progress(0.0, text="Starting deep crawl…")
        progress_text = st.empty()
        completed_ref = {"n": 0}

        def _on_progress(done, total):
            completed_ref["n"] = done
            pct = done / total if total else 0
            progress_bar.progress(pct, text=f"Deep crawling… {done}/{total}")
            progress_text.caption(f"{done} of {total} pages crawled")

        with st.spinner(f"Deep crawling {len(sources)} pages via {tier}…"):
            crawled = crawl_sources(
                sources,
                max_workers=min(threads, 5),
                tier=tier,
                progress_callback=_on_progress,
            )

        progress_bar.empty()
        progress_text.empty()

        if crawled:
            existing = st.session_state.get(scraped_key, {})
            existing.update(crawled)
            st.session_state[scraped_key] = existing

            # Persist every successfully crawled URL into the seeds DB.
            saved_count = 0
            for url, text in crawled.items():
                src_title = next((s.get("title","") for s in sources if s.get("link")==url), "")
                try:
                    seed_db.add_seed(url, src_title)
                    sid = seed_db.get_seed_by_url(url)
                    if sid:
                        seed_db.mark_crawled(sid["id"], status_code=200, content=text)
                        saved_count += 1
                except Exception:
                    pass

            st.success(
                f"✅ Deep crawled {len(crawled)}/{len(sources)} pages "
                f"({sum(len(v) for v in crawled.values()):,} total chars). "
                f"{saved_count} saved to **Seed Manager** (sidebar). "
                "HTML archived to `investigations/crawled/<hash>/`. "
                "Use **Re-summarize** below to regenerate findings with this richer content."
            )
        else:
            st.warning("No pages could be deep crawled. Check Tor connectivity.")


# ---------------------------------------------------------------------------
# Loaded investigation view
# ---------------------------------------------------------------------------

if "loaded_investigation" in st.session_state and not run_button:
    inv    = st.session_state["loaded_investigation"]
    inv_id = inv.get("id")

    st.info(f"📂 **{inv['query']}** — {inv['timestamp'][:16]}")

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
            st.success("Updated.")
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
        st.download_button("📄 Download PDF Report", data=pdf_bytes,
                            file_name=f"robin_investigation_{inv['timestamp'][:10]}.pdf",
                            mime="application/pdf", key="pdf_dl_loaded")

    # Deep crawl for loaded investigation sources
    if inv["sources"]:
        st.divider()
        st.markdown("### 🕷️ Deep Crawl")
        st.caption(
            "Fetch richer content directly from the sources referenced in this investigation, "
            "then regenerate the findings with that deeper context."
        )
        _deep_crawl_sources_section(
            sources=inv["sources"],
            scraped_key="loaded_inv_scraped",
            query_text=inv.get("refined_query", inv.get("query", "")),
            section_label="investigation sources",
        )

        # ── Re-summarize THIS investigation ──────────────────────────────────
        st.divider()
        st.markdown("### 🔁 Re-summarize this investigation")
        st.caption(
            "Regenerate the findings using the loaded investigation's sources "
            "and the currently selected model/preset. Deep-crawled content (if any) "
            "is used automatically; otherwise sources are re-scraped on the fly."
        )
        if st.button(
            "🔁 Re-summarize this investigation",
            use_container_width=True,
            key=f"resummarize_loaded_{inv_id}",
        ):
            with st.status("✍️ Re-summarizing loaded investigation…", expanded=True) as rs_status:
                st.write("🔄 Loading LLM…")
                try:
                    llm = get_llm(model)
                except Exception as e:
                    rs_status.update(label="❌ Failed", state="error")
                    _render_pipeline_error("load the selected LLM", e)

                # Prefer deep-crawled content; fall back to a fresh scrape of
                # the loaded investigation's sources so this button always works.
                content_dict = st.session_state.get("loaded_inv_scraped") or {}
                if not content_dict:
                    st.write("📜 Scraping loaded investigation sources…")
                    content_dict = cached_scrape_multiple(
                        inv["sources"], threads, max_content_chars,
                    )
                    st.write(f"&nbsp;&nbsp;&nbsp;→ {len(content_dict)} pages scraped")
                else:
                    st.write(
                        f"📦 Using {len(content_dict)} deep-crawled pages "
                        f"({sum(len(v) for v in content_dict.values()):,} chars)."
                    )

                rs_query = inv.get("refined_query") or inv.get("query", "")
                st.write("✍️ Streaming new summary…")

                # Stream into a temporary slot so the user can watch the
                # tokens arrive. Once streaming finishes we replace this
                # block with the full _render_findings() output (Markdown +
                # download buttons) — using a SECOND placeholder defined
                # below as the explicit render target.
                streaming_slot = st.empty()
                with streaming_slot.container():
                    st.subheader(":red[🔎 Findings (re-summarized)]", anchor=None, divider="gray")
                    summary_slot = st.empty()

                new_summary = _run_summary_stage(
                    llm, rs_query, content_dict,
                    selected_preset, custom_instructions, summary_slot,
                    system_prompt_override=selected_system_prompt,
                )
                rs_status.update(label="✅ Re-summarization complete", state="complete", expanded=False)

            # Clear the streaming placeholder so the final findings render
            # below replaces it cleanly (no duplicate headings).
            streaming_slot.empty()

            # Save as a NEW investigation linked to the same query so the
            # original record stays intact for comparison.
            new_inv_id = inv_db.save_investigation(
                query=inv.get("query", ""),
                refined_query=rs_query,
                model=model,
                preset_label=selected_preset_label,
                sources=inv["sources"],
                summary=new_summary,
            )
            st.success(
                f"✅ Saved as new investigation #{new_inv_id}. "
                "The original investigation is unchanged."
            )

            inv_dict_for_pdf = {
                "query":         inv.get("query", ""),
                "refined_query": rs_query,
                "model":         model,
                "preset":        selected_preset_label,
                "status":        "active",
                "tags":          "",
                "timestamp":     datetime.now().isoformat(),
                "sources":       inv["sources"],
                "summary":       new_summary,
            }
            # Pass an explicit local placeholder — the module-level
            # `findings_placeholder` doesn't exist yet at this point in
            # script execution (it's defined below the loaded-inv block).
            _render_findings(new_summary, inv_dict_for_pdf, target=st.empty())

    if st.button("✖ Clear"):
        del st.session_state["loaded_investigation"]
        st.session_state.pop("loaded_inv_scraped", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Pipeline placeholders
# ---------------------------------------------------------------------------

status_slot          = st.empty()
notes_placeholder    = st.empty()
sources_placeholder  = st.empty()
findings_placeholder = st.empty()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

if run_button and query:
    st.session_state.pop("loaded_investigation", None)
    for k in ["refined", "results", "filtered", "scraped", "streamed_summary", "last_inv_id"]:
        st.session_state.pop(k, None)

    with st.status("🔄 Running investigation pipeline…", expanded=True) as pipeline_status:

        st.write("🔄 Loading LLM…")
        try:
            llm = get_llm(model)
        except Exception as e:
            pipeline_status.update(label="❌ Pipeline failed", state="error", expanded=True)
            _render_pipeline_error("load the selected LLM", e)

        st.write("✏️ Refining query…")
        try:
            st.session_state.refined = refine_query(llm, query)
            st.write(f"&nbsp;&nbsp;&nbsp;→ `{st.session_state.refined}`")
        except Exception as e:
            pipeline_status.update(label="❌ Pipeline failed", state="error", expanded=True)
            _render_pipeline_error("refine the query", e)

        st.write("🔍 Searching dark web (results pre-scored by keyword relevance)…")
        st.session_state.results = cached_search_results(st.session_state.refined, threads)
        if len(st.session_state.results) > max_results:
            st.session_state.results = st.session_state.results[:max_results]
        st.write(f"&nbsp;&nbsp;&nbsp;→ {len(st.session_state.results)} unique results")

        num_batches = (len(st.session_state.results) + 24) // 25
        st.write(f"🗂️ Filtering results ({num_batches} batch{'es' if num_batches > 1 else ''})…")
        st.session_state.filtered = filter_results(
            llm, st.session_state.refined, st.session_state.results
        )
        if len(st.session_state.filtered) > max_scrape:
            st.session_state.filtered = st.session_state.filtered[:max_scrape]
        st.write(f"&nbsp;&nbsp;&nbsp;→ {len(st.session_state.filtered)} selected")

        st.write("📜 Scraping content…")
        st.session_state.scraped = cached_scrape_multiple(
            st.session_state.filtered, threads, max_content_chars
        )
        st.write(f"&nbsp;&nbsp;&nbsp;→ {len(st.session_state.scraped)} pages scraped")

        st.write("✍️ Generating summary…")
        with findings_placeholder.container():
            st.subheader(":red[🔎 Findings]", anchor=None, divider="gray")
            summary_slot = st.empty()

        st.session_state.streamed_summary = _run_summary_stage(
            llm, query, st.session_state.scraped,
            selected_preset, custom_instructions, summary_slot,
            system_prompt_override=selected_system_prompt,
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

    # Sources + deep crawl section
    with sources_placeholder.container():
        with st.expander(
            f"🔗 Sources ({len(st.session_state.filtered)} results)", expanded=False
        ):
            for i, item in enumerate(st.session_state.filtered, 1):
                st.markdown(f"{i}. [{item.get('title','Untitled')}]({item.get('link','')})")

        # ── Deep Crawl block lives right below sources ──
        st.divider()
        st.markdown("### 🕷️ Deep Crawl")
        st.caption(
            "Go deeper than Robin's standard scrape — fetch full JS-rendered content "
            "from individual sources or all at once, then regenerate the findings with "
            "richer context."
        )
        _deep_crawl_sources_section(
            sources=st.session_state.filtered,
            scraped_key="scraped",
            query_text=st.session_state.refined,
            section_label="result sources",
        )

    # Findings
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
# Re-summarize — reuses cached scraped data (updated by deep crawl)
#
# Only fires for a fresh-run context. When a past investigation is loaded,
# the loaded view has its own Re-summarize button that operates on that
# investigation's sources — without this guard, the bottom button would
# operate on stale fresh-run data even while a different investigation is
# being viewed.
# ---------------------------------------------------------------------------

if (
    st.session_state.get("scraped")
    and not run_button
    and "loaded_investigation" not in st.session_state
):
    st.divider()
    st.caption(
        "🔁 **Re-summarize** — change model/preset above, or after a deep crawl, "
        "to regenerate findings with updated content."
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
                system_prompt_override=selected_system_prompt,
            )
            rs_status.update(label="✅ Re-summarization complete", state="complete", expanded=False)

        new_inv_id = inv_db.save_investigation(
            query=original_query,
            refined_query=st.session_state.get("refined", original_query),
            model=model,
            preset_label=selected_preset_label,
            sources=st.session_state.get("filtered", []),
            summary=st.session_state.streamed_summary,
        )

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