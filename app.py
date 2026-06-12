"""
KV Cache Demo — single-file Streamlit app.

Show the value of KV-cache reuse across a multi-turn conversation
against any OpenAI-compatible endpoint.

Run:
    pip install streamlit openai pandas
    streamlit run app.py
"""
import json
import os
import secrets
import time

import altair as alt
import pandas as pd
import streamlit as st
from openai import OpenAI

# ============================================================
# Env defaults
# ============================================================
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

LANG_OPTIONS = ["CN", "EN", "JA", "KO", "FR", "DE"]
LANG_NAMES = {
    "CN": "Chinese (Simplified)",
    "EN": "English",
    "JA": "Japanese",
    "KO": "Korean",
    "FR": "French",
    "DE": "German",
}

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="KV Cache Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Snapshot data — hardcoded vLLM release v0.6.4.post1 (hotfix)
# Inlined so the demo runs offline / without GitHub API access.
# ============================================================
SNAPSHOT_RELEASE = {
    "tag": "v0.6.4.post1",
    "is_hotfix": True,
    "major_minor": "0.6",
    "published_at": "2025-01-15",
    "raw_body": (
        "## Highlights\n"
        "This is a patch release on top of v0.6.4, addressing three regressions "
        "introduced in v0.6.4.\n\n"
        "### Bug Fixes\n"
        "* Fix CUDA OOM in chunked prefill when prefix caching is enabled (#PR1) — @alice-dev\n"
        "* Fix incorrect token sampling when top_k=1 with temperature=0 in V1 engine (#PR2) — @bob-llm\n"
        "* Fix dist-kv race condition during engine startup (#PR3) — @carol-infer\n\n"
        "### Known Issues\n"
        "* V1 engine still slower than V0 on small batches\n"
        "* Some quantization kernels not yet migrated to V1\n"
    ),
    "prs": [
        {
            "number": "PR1",
            "title": "Fix CUDA OOM in chunked prefill when prefix caching is enabled",
            "author": "alice-dev",
            "body": (
                "**Problem:** When prefix caching is on and a request's prefix partially "
                "matches a cached block, the V1 engine over-allocates GPU memory for the "
                "new blocks because it doesn't reuse the matched prefix's KV tensor slots "
                "in time. On H100 with long prefixes, this can OOM mid-prefill.\n\n"
                "**Fix:** Pre-compute the matched prefix's tensor slot count at request "
                "admission time and subtract from the new allocation budget. Adds a new "
                "field `prefix_slot_reservation` in `SchedulerOutput`.\n\n"
                "**Impact:** ~5% prefill throughput regression on cache-hit-heavy "
                "workloads, but eliminates the OOM crash. Test coverage added in "
                "`tests/kernels/test_prefix_cache_alloc.py`."
            ),
        },
        {
            "number": "PR2",
            "title": "Fix incorrect token sampling when top_k=1 with temperature=0 in V1 engine",
            "author": "bob-llm",
            "body": (
                "**Problem:** With `top_k=1, temperature=0`, the V1 engine was returning "
                "the second-highest logit instead of the argmax in ~0.1% of cases due to "
                "a floating-point tiebreak bug in `flashinfer.sampling.top_k_top_p`.\n\n"
                "**Fix:** Use `torch.argmax` directly when temperature==0 and top_k<=1, "
                "bypass flashinfer path.\n\n"
                "**Impact:** Correctness fix; performance neutral. Affects all V1 users "
                "with greedy decoding on supported models."
            ),
        },
    ],
    "authors": [
        {
            "login": "alice-dev",
            "name": "Alice Chen",
            "company": "Anthropic (Inference Platform)",
            "bio": "Production LLM serving infra. Previously: PyTorch core, CUDA kernel "
            "optimization at NVIDIA.",
            "recent_activity": [
                "vllm #8901 — Prefix cache memory budget",
                "vllm #8750 — V1 scheduler refactor (co-author)",
                "pytorch #1234 — SDPA backward pass (commented)",
                "Blog: 'KV cache is the new memory hierarchy' (2024-12)",
            ],
        },
        {
            "login": "bob-llm",
            "name": "Bob Martinez",
            "company": "Mistral AI",
            "bio": "Inference engineer. Sampling correctness, speculative decoding, "
            "low-latency serving.",
            "recent_activity": [
                "vllm #8840 — Sampling fix (this PR)",
                "vllm #8700 — Speculative decoding benchmark",
                "mistral-inference #200 — Continuous batching",
                "PyData 2024 talk: 'Why your LLM sampler is wrong'",
            ],
        },
    ],
}

# ============================================================
# Scripted user messages (6 turns)
# ============================================================
SCRIPT = [
    "Look at the release data above. Is this an x.y minor release or an x.y.z patch/hotfix? "
    "If it's a hotfix, analyze the likely reasons behind it — what kind of bugs justify a post-release?",
    "Pick the first PR from the release notes. Walk me through: what it changed, what the "
    "impact surface is, and any potential risks.",
    "Now analyze PR #1's author based on the GitHub profile data above. What's their background, "
    "and what does their recent activity tell you about where they focus their work?",
    "Move on to the second PR. Same deep-dive format: change, impact, risks.",
    "Analyze PR #2's author the same way: background and recent activity signals.",
    "Wrap up with a ~200-word synthesis: what does this release tell you about vLLM's current "
    "direction, the risk surface of recent changes, and the engineering style of the contributor set?",
]

# ============================================================
# Default system prompt (persona + reference data block)
# ~1.4K tokens — exceeds the 1K threshold for auto KV caching
# ============================================================
DEFAULT_SYSTEM_PROMPT = (
    "You are a senior LLM inference engineer with deep production experience at "
    "large training/inference orgs. You review releases, PRs, and engineering "
    "artifacts with the rigor of a staff engineer at an AI infrastructure company.\n\n"
    "# Style\n"
    "- Direct, specific, no hedging. If you'd say 'consider' or 'perhaps', just say what to do.\n"
    "- Reference concrete details (PR numbers, file names, API signatures) when relevant.\n"
    "- Use bullet points for lists.\n"
    "- For risk analysis, always include likelihood + impact.\n"
    "- Keep each response focused: answer the asked question, don't recap.\n\n"
    "# Framework\n"
    "For each PR: (1) what changed, (2) impact surface, (3) potential risks with likelihood/impact.\n"
    "For each author: (1) signal from bio/company, (2) recent activity themes, (3) inferred focus.\n\n"
    "# Reference data\n"
    "The release data below is the only authoritative source for the version, PRs, and author "
    "profiles. Use it as ground truth.\n\n"
    "```json\n"
    + json.dumps(SNAPSHOT_RELEASE, indent=2)
    + "\n```"
)

# ============================================================
# Helpers
# ============================================================
@st.cache_resource
def get_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key or "EMPTY")


def call_llm(client: OpenAI, model: str, messages: list, temperature: float = 1.0):
    """Stream a chat completion, return (content, usage, ttft_ms, e2e_ms)."""
    t0 = time.perf_counter()
    ttft = None
    parts: list[str] = []
    usage = None
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        temperature=temperature,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            parts.append(chunk.choices[0].delta.content)
        if chunk.usage:
            usage = chunk.usage
    e2e = (time.perf_counter() - t0) * 1000
    return "".join(parts), usage, (ttft if ttft is not None else e2e), e2e


def extract_usage(usage) -> dict:
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "cached_tokens": None}
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    cached = None
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
    return {"prompt_tokens": pt or None, "completion_tokens": ct or None, "cached_tokens": cached}


# ============================================================
# Session state
# ============================================================
_DEFAULTS = {
    "messages": [],
    "turns": [],
    "running": False,
    "next_turn": 0,
    "ms_per_token_turn1": None,
    "stop": False,
    "finished": False,
    "last_error": None,
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ============================================================
# Sidebar — endpoint config
# ============================================================
with st.sidebar:
    st.title("⚙️ Config")
    base_url = st.text_input(
        "API Base URL (OpenAI-compatible, must end in /v1)",
        value=DEFAULT_BASE_URL,
    )
    api_key = st.text_input("API Key", type="password", value=DEFAULT_API_KEY)
    model = st.text_input("Model", value=DEFAULT_MODEL)
    language = st.selectbox("Output language", LANG_OPTIONS, index=0,
                            help="Appended to the system prompt at run start.")
    with st.expander("System prompt (editable)", expanded=False):
        sys_prompt = st.text_area("system", value=DEFAULT_SYSTEM_PROMPT, height=200, label_visibility="collapsed")
    max_turns = st.slider("Max turns to run", 1, len(SCRIPT), len(SCRIPT))
    temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1,
                                  help="Default 1.0 — works for models that only accept 1 (e.g. some o-series / gpt-5).")

    st.divider()
    start_btn = st.button("Start", type="primary", disabled=st.session_state.running)
    stop_btn = st.button("Stop", disabled=not st.session_state.running)

    if start_btn:
        marker = f"# Session marker (ignore, only invalidates prior cache): {secrets.token_hex(4)}\n\n"
        full_system = marker + sys_prompt + f"\n\n# Output language\nRespond in {LANG_NAMES[language]}."
        st.session_state.messages = [{"role": "system", "content": full_system}]
        st.session_state.turns = []
        st.session_state.running = True
        st.session_state.next_turn = 0
        st.session_state.ms_per_token_turn1 = None
        st.session_state.finished = False
        st.session_state.stop = False
        st.session_state.last_error = None
        st.rerun()
    if stop_btn:
        st.session_state.stop = True
        st.session_state.running = False

# ============================================================
# Main — chat left, metrics right
# ============================================================
left, right = st.columns([3, 2])

with left:
    st.subheader("💬 Conversation")
    chat_box = st.container(height=620, border=True)
    with chat_box:
        if not st.session_state.messages and not st.session_state.running:
            st.info("Configure the endpoint in the sidebar and press ▶ Start.")
        for msg in st.session_state.messages:
            if msg["role"] == "system":
                continue
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if st.session_state.last_error:
            st.error(st.session_state.last_error)

with right:
    st.subheader("📊 Live metrics")
    if not st.session_state.turns:
        st.caption("No data yet. Press ▶ Start.")
    else:
        latest = st.session_state.turns[-1]
        c1, c2 = st.columns(2)
        c1.metric(
            "Prompt tokens",
            f"{latest['prompt_tokens']:,}" if latest["prompt_tokens"] else "—",
        )
        c2.metric(
            "Cached tokens",
            f"{latest['cached_tokens']:,}" if latest["cached_tokens"] is not None else "—",
        )
        c3, c4 = st.columns(2)
        c3.metric(
            "KV cache reuse",
            f"{latest['kv_reuse_pct']:.1f}%" if latest["kv_reuse_pct"] is not None else "N/A",
        )
        if latest.get("speedup_display") is not None:
            c4.metric(
                "TTFT speedup",
                f"{latest['pct_display']:+.0f}%",
                delta=f"{latest['speedup_display']:.2f}x",
            )
        else:
            c4.metric("TTFT speedup", "N/A")

        st.caption("**TTFT per turn (ms)** — solid bars = actual (cached) · "
                   "dashed line = no-cache estimate (baseline)")
        chart_rows = []
        for t in st.session_state.turns:
            chart_rows.append({"Turn": t["turn"], "Actual (cached)": round(t["ttft_ms"] or 0)})
        if chart_rows:
            actual_df = pd.DataFrame(chart_rows).set_index("Turn")
            estimate_df = pd.DataFrame(
                [
                    {"Turn": t["turn"], "No-cache estimate": round(t["no_cache_estimate_ms"])}
                    for t in st.session_state.turns
                    if t["no_cache_estimate_ms"] is not None
                ]
            ).set_index("Turn")
            bars = alt.Chart(actual_df.reset_index()).mark_bar(size=36, color="#6366f1").encode(
                x=alt.X("Turn:O", title="Turn"),
                y=alt.Y("Actual (cached):Q", title="TTFT (ms)"),
                tooltip=["Turn", alt.Tooltip("Actual (cached):Q", title="Actual (ms)")],
            )
            layers = [bars]
            if not estimate_df.empty:
                line_df = estimate_df.reset_index()
                line = alt.Chart(line_df).mark_line(
                    strokeDash=[6, 4], color="#f97316", strokeWidth=2.5
                ).encode(x="Turn:O", y=alt.Y("No-cache estimate:Q"))
                points = alt.Chart(line_df).mark_point(
                    color="#f97316", size=70, filled=True
                ).encode(x="Turn:O", y=alt.Y("No-cache estimate:Q"))
                layers.extend([line, points])
            st.altair_chart(alt.layer(*layers).properties(height=280), use_container_width=True)

        st.caption("**Per-turn detail**")
        rows = []
        for t in st.session_state.turns:
            rows.append(
                {
                    "Turn": t["turn"],
                    "TTFT": f"{t['ttft_ms']:.0f} ms" if t["ttft_ms"] else "—",
                    "Prompt": f"{t['prompt_tokens']:,}" if t["prompt_tokens"] else "—",
                    "Cached": f"{t['cached_tokens']:,}" if t["cached_tokens"] is not None else "—",
                    "Reuse": f"{t['kv_reuse_pct']:.1f}%" if t["kv_reuse_pct"] is not None else "—",
                    "Speedup": f"{t['speedup_display']:.2f}x" if t.get("speedup_display") else "—",
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)

        if st.session_state.finished and st.session_state.turns:
            last = st.session_state.turns[-1]
            st.success(
                f"Run complete · {len(st.session_state.turns)} turns · "
                f"last turn reuse {last['kv_reuse_pct']:.0f}%"
                if last["kv_reuse_pct"] is not None
                else f"Run complete · {len(st.session_state.turns)} turns"
            )

        st.markdown(
            "<p style='font-size:11px; color:#6b7280; font-style:italic; margin-top:1.5em;'>"
            "▎&nbsp; The no-cache number is an estimate extrapolated from the first turn's cold "
            "ms-per-token. Real no-cache numbers depend on backend load, but the direction holds."
            "</p>",
            unsafe_allow_html=True,
        )

# ============================================================
# Run loop — one turn per rerun (Streamlit's standard pattern)
# ============================================================
if st.session_state.running and not st.session_state.stop:
    if st.session_state.next_turn < min(max_turns, len(SCRIPT)):
        turn_idx = st.session_state.next_turn
        st.session_state.messages.append({"role": "user", "content": SCRIPT[turn_idx]})

        try:
            client = get_client(base_url, api_key)
            with st.spinner(f"Turn {turn_idx + 1} / {min(max_turns, len(SCRIPT))}"):
                content, usage, ttft, e2e = call_llm(client, model, st.session_state.messages, temperature)
            st.session_state.messages.append({"role": "assistant", "content": content})

            u = extract_usage(usage)
            rec = {
                "turn": turn_idx + 1,
                "ttft_ms": ttft,
                "e2e_ms": e2e,
                "prompt_tokens": u["prompt_tokens"],
                "cached_tokens": u["cached_tokens"],
                "completion_tokens": u["completion_tokens"],
                "kv_reuse_pct": None,
                "ms_per_token": None,
                "no_cache_estimate_ms": None,
                "speedup": None,
                "speedup_pct": None,
            }
            if u["prompt_tokens"]:
                if turn_idx == 0:
                    st.session_state.ms_per_token_turn1 = ttft / u["prompt_tokens"]
                mspt = st.session_state.ms_per_token_turn1
                if u["cached_tokens"] is not None:
                    rec["kv_reuse_pct"] = (u["cached_tokens"] / u["prompt_tokens"]) * 100
                if mspt and mspt > 0:
                    rec["ms_per_token"] = mspt
                    rec["no_cache_estimate_ms"] = mspt * u["prompt_tokens"]
                    if ttft > 0:
                        rec["speedup"] = rec["no_cache_estimate_ms"] / ttft
                        rec["speedup_display"] = round(rec["speedup"], 2)
                        rec["pct_display"] = (rec["speedup_display"] - 1) * 100
            st.session_state.turns.append(rec)
            st.session_state.next_turn += 1
        except Exception as e:
            st.session_state.last_error = f"{type(e).__name__}: {e}"
            st.session_state.running = False
            st.session_state.stop = True
        st.rerun()
    else:
        st.session_state.running = False
        st.session_state.finished = True
        st.rerun()
elif st.session_state.stop:
    st.session_state.stop = False
