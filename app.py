"""
Enterprise-level Streamlit: Text -> Math Problem Solver using ChatGroq (GROQ) + local safe evaluation.

Features:
- User enters GROQ API key and Model name (default: llama-3.3-70b-versatile).
- Sends prompt to ChatGroq (langchain_groq.ChatGroq).
- Parses the model answer and extracts numeric expression(s).
- Safely evaluates math expressions using sympy.
- Displays model explanation, local computation, difference/validation.
- Conversation history, export, caching, logging, and polished UI.

NOTE: Replace ChatGroq usage with your organization's client if API surface differs.
"""

import streamlit as st
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import time
import json
import logging
import re
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
import math

# Safe math evaluation via sympy
import sympy as sp

load_dotenv()

# If you use langchain_groq import, keep it optional for local dev
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None  # will fail only when trying to call API

# ---------------------------
# Logging & utility config
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("math-solver-app")

# ---------------------------
# Data classes
# ---------------------------

@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    text: str
    timestamp: str

@dataclass
class Conversation:
    messages: List[Message]

    def append(self, role: str, text: str):
        self.messages.append(Message(role=role, text=text, timestamp=datetime.utcnow().isoformat()))

    def as_list(self):
        return [asdict(m) for m in self.messages]

# ---------------------------
# Utility functions
# ---------------------------
def now_iso():
    return datetime.utcnow().isoformat()

def sanitize_api_key(key: str) -> str:
    return key.strip()

def init_session_state():
    if "conv" not in st.session_state:
        st.session_state.conv = Conversation(messages=[])
    if "model_name" not in st.session_state:
        st.session_state.model_name = "qwen/qwen3-32b"
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""
    if "last_run_meta" not in st.session_state:
        st.session_state.last_run_meta = {}

# extract numeric expressions from text (simple heuristic)
EXPR_REGEX = re.compile(r"([0-9+\-*/().\s^%]+)")

def extract_expressions(text: str) -> List[str]:
    """
    Find likely arithmetic expressions in text.
    This is intentionally conservative — sympy parsing will handle most cases.
    """
    candidates = []
    for m in EXPR_REGEX.finditer(text):
        expr = m.group(1).strip()
        # skip if too short or just punctuation
        if len(expr) >= 1 and any(ch.isdigit() for ch in expr):
            # replace ^ with ** for pythonic exponent, % -> /100
            expr_py = expr.replace("^", "**").replace("%", "/100")
            # basic cleanup: collapse multiple spaces
            expr_py = re.sub(r"\s+", " ", expr_py).strip()
            candidates.append(expr_py)
    # unique preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique

def safe_eval_sympy(expr: str) -> Dict[str, Any]:
    """
    Safely parse and evaluate expression using sympy.
    Returns dict with 'success', 'value', 'error', 'sympy_repr'.
    """
    try:
        expr_sym = sp.sympify(expr, evaluate=True)
        numeric = None
        try:
            numeric = float(sp.N(expr_sym))
        except Exception:
            numeric = None
        return {
            "success": True,
            "sympy": str(expr_sym),
            "numeric": numeric,
            "repr": repr(expr_sym)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ---------------------------
# Improved extraction for final numeric answer
# ---------------------------

def extract_final_number(text: str) -> Optional[float]:
    """
    Try to extract the final numeric answer from model text.
    Priority:
      1. Look for explicit labels (Answer:, Final:, Result:, Final answer:)
      2. Look for numbers in last few lines
      3. Fallback: last numeric token in the whole text
    Returns float or None.
    """
    if not text:
        return None

    # 1) explicit labels (case-insensitive)
    explicit_patterns = [
        r"final answer\s*[:\-=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        r"answer\s*[:\-=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        r"result\s*[:\-=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        r"final\s*[:\-=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    ]
    for p in explicit_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass

    # 2) look for numbers at end of lines that are likely final sentences
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines[-3:]):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ln)
        if nums:
            try:
                return float(nums[-1])
            except Exception:
                continue

    # 3) fallback to last numeric token in entire text
    all_nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if all_nums:
        try:
            return float(all_nums[-1])
        except Exception:
            return None

    return None

# ---------------------------
# Numeric extraction & auto-correction helpers
# ---------------------------

def safe_extract_numeric(text_or_obj) -> Optional[float]:
    """
    Robust numeric extractor:
    - Accepts numeric types, objects with .content, or strings.
    - Returns float or None.
    """
    if text_or_obj is None:
        return None
    if isinstance(text_or_obj, (int, float)):
        return float(text_or_obj)
    # AIMessage-like
    if hasattr(text_or_obj, "content"):
        text = str(text_or_obj.content)
    else:
        text = str(text_or_obj)
    text = text.strip()
    # direct conversion if entire string is numeric
    try:
        return float(text)
    except Exception:
        pass
    # try explicit final extraction (prefer explicit markers)
    explicit = extract_final_number(text)
    if explicit is not None:
        return explicit
    # fallback to first numeric token
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None

def request_correction_and_replace(client, orig_prompt: str, model_text: str,
                                   final_number: Optional[float], local_numeric: float,
                                   temperature: float, max_tokens: int):
    """
    Silent auto-correction (Option B):
    - Ask model to recompute the whole solution step-by-step and provide a corrected full explanation.
    Returns (corrected_model_text, corrected_final_number, meta)
    """
    correction_system = "You are a careful math solver. Recompute and correct any arithmetic mistakes."
    correction_prompt = (
        "Your previous answer may be incorrect. Recompute the entire problem step-by-step and ensure all arithmetic is correct. "
        "Provide intermediate expressions and then a final numeric answer. At the end include a single line like 'Final answer: <number>'.\n\n"
        "Original problem:\n"
        f"{orig_prompt}\n\n"
        "Do not include unrelated commentary."
    )
    try:
        corrected_text, meta = ask_groq(
            client,
            user_prompt=correction_prompt,
            system_prompt=correction_system,
            temperature=0.0,
            max_tokens=max_tokens
        )
    except Exception as e:
        logger.exception("Correction attempt failed: %s", e)
        return model_text, final_number, {"correction_error": str(e)}

    corrected_final = extract_final_number(corrected_text)
    return corrected_text, corrected_final if corrected_final is not None else final_number, {"raw_correction": meta}

# ---------------------------
# GROQ / ChatGroq wrapper
# ---------------------------

def create_groq_client(api_key: str, model_name: str, **kwargs):
    """
    Create ChatGroq client.
    """
    if ChatGroq is None:
        raise RuntimeError("ChatGroq client not available. Install langchain_groq or provide a replacement.")
    client = ChatGroq(api_key=api_key, model=model_name, **kwargs)
    return client

def ask_groq(client, user_prompt: str, system_prompt: Optional[str] = None,
             temperature: float = 0.0, max_tokens: int = 1024):
    """
    Call ChatGroq using the recommended chat API (invoke with messages).
    Returns (text, metadata).
    """
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=user_prompt))

    response = client.invoke(
        messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Ensure we return plain string content and raw metadata
    try:
        content = response.content
    except Exception:
        content = str(response)
    return content, {"raw": response}

# ---------------------------
# Streamlit UI
# ---------------------------

init_session_state()

st.set_page_config(page_title="Enterprise Math Solver — GROQ + Streamlit", layout="wide", initial_sidebar_state="expanded")

# custom CSS for attractive UI
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a, #001f3f);
        color: #e6eef8;
    }
    .title {
        font-size:28px;
        font-weight:700;
        color: #fff;
    }
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        border-radius: 12px;
        padding: 16px;
    }
    .muted {
        color: #a5b4cc;
    }
    pre {
        white-space: pre-wrap;
        word-break: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
with st.container():
    col1, col2 = st.columns([9, 1])
    with col1:
        st.markdown("<div class='title'>🧮 Enterprise Math Problem Solver — GROQ + Streamlit</div>", unsafe_allow_html=True)
        st.markdown("<div class='muted'>Text → Math solver. Model explains the steps and app verifies answers locally.</div>", unsafe_allow_html=True)
    with col2:
        st.image("https://assets.codepen.io/285131/logo.png", width=48) if False else None

# Sidebar: API key and model config
with st.sidebar:
    st.header("Configuration")
    st.markdown("Enter GROQ credentials and model. You can save to session for temporary use.")
    api_key = st.text_input("GROQ API Key", value=st.session_state.groq_api_key, type="password", placeholder="sk-... (enter your key)")
    model_name = st.text_input("Model Name", value=st.session_state.model_name, placeholder="llama-3.3-70b-versatile")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    max_tokens = st.slider("Max tokens (response)", min_value=128, max_value=8192, value=1024, step=128)
    persist_key = st.checkbox("Store API Key in session (not persistent)", value=False)
    st.markdown("---")
    st.markdown("Enterprise options:")
    st.checkbox("Enable verbose logging to console", value=False, key="verbose_logging")
    st.button("Clear conversation", on_click=lambda: st.session_state.conv.messages.clear())

    # persist to session if requested
    if persist_key:
        st.session_state.groq_api_key = sanitize_api_key(api_key)
    st.session_state.model_name = model_name or st.session_state.model_name

# Main panel
left, right = st.columns([3, 2])

with left:
    st.subheader("Enter Math Word Problem or Expression")
    prompt = st.text_area("Problem (text)", height=180, placeholder="e.g. Two pipes can fill a tank in 3 and 5 hours. How long together? Show steps and final numeric answer.")
    advanced = st.expander("Advanced options")
    with advanced:
        show_extraction = st.checkbox("Show extracted numeric expressions", value=True)
        run_local_eval = st.checkbox("Also evaluate extracted expressions locally (sympy)", value=True)

    submit = st.button("Solve ✅")

    if submit:
        if not api_key:
            st.error("GROQ API Key required. Enter it in the sidebar.")
        else:
            st.session_state.conv.append("user", prompt)
            # instantiate client
            try:
                client = create_groq_client(api_key=api_key.strip(), model_name=st.session_state.model_name)
            except Exception as e:
                st.error(f"Failed to create GROQ client: {e}")
                logger.exception(e)
                client = None

            # stronger system prompt to always include final marker
            system_prompt = (
                """
                You are an expert math tutor. Provide a very short solution:
                - One-line plan, 
                - Minimal intermediate calculations (one per line), 
                - Last line exactly: Final answer: <number> <currency/unit if applicable>.
                Use exact decimals when needed. Keep output ≤5 lines.

                """
            )

            try:
                # Get model response
                model_text, meta = ask_groq(client, user_prompt=prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens)
                st.session_state.conv.append("assistant", model_text)
            except Exception as e:
                st.error(f"Model call failed: {e}")
                logger.exception(e)
                model_text = f"Model call failed: {e}"
                st.session_state.conv.append("assistant", model_text)

            # Show model explanation
            st.markdown("### Model explanation")
            st.markdown(f"<div class='card'><pre>{model_text}</pre></div>", unsafe_allow_html=True)

            # Extract expressions from model text and user prompt
            exprs_from_prompt = extract_expressions(prompt)
            exprs_from_model = extract_expressions(model_text)
            if show_extraction:
                st.markdown("**Extracted expressions**")
                st.write("From user prompt:", exprs_from_prompt)
                st.write("From model reply:", exprs_from_model)

            # Choose candidate expression(s) to evaluate:
            candidates = exprs_from_model + [e for e in exprs_from_prompt if e not in exprs_from_model]
            candidates = list(dict.fromkeys(candidates))  # unique preserve order

            # Local evaluation
            eval_results = []
            if run_local_eval and candidates:
                st.markdown("### Local evaluation (sympy)")
                for i, expr in enumerate(candidates):
                    res = safe_eval_sympy(expr)
                    eval_results.append({"expression": expr, **res})
                    if res.get("success"):
                        st.write(f"Expression {i+1}: `{expr}` -> sympy: `{res['sympy']}` numeric: `{res['numeric']}`")
                    else:
                        st.write(f"Expression {i+1}: `{expr}` -> Error: {res.get('error')}")
                # compare model numeric answers (try to find numeric in model text)
                numeric_finds = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", model_text)
                if numeric_finds:
                    st.markdown("**Numeric values found in model reply**")
                    st.write(numeric_finds[:50])
            elif run_local_eval:
                st.info("No arithmetic expressions extracted for local evaluation.")

            # ---------------------------
            # Validation: compare model's final numeric answer to local evaluation
            # ---------------------------
            st.markdown("### Validation / Verdict")

            final_number = extract_final_number(model_text)

            if final_number is None:
                st.warning("Could not detect a final numeric answer in the model reply automatically.")
            else:
                st.write("Model final numeric ->", final_number)

                # prefer the LAST numeric evaluation (likely the final result)
                numeric_candidates = [r["numeric"] for r in eval_results if r.get("numeric") is not None]
                numeric_local = numeric_candidates[-1] if numeric_candidates else None

                # heuristic: if numeric_local missing, try to find eval that matches final_number
                if numeric_local is None and eval_results and final_number is not None:
                    closest = None
                    for r in eval_results:
                        if r.get("numeric") is not None:
                            if abs(r["numeric"] - final_number) < 1e-6:
                                closest = r["numeric"]
                                break
                    if closest is not None:
                        numeric_local = closest

                if numeric_local is not None:
                    st.write("Local numeric ->", numeric_local)
                    diff = abs(final_number - numeric_local)
                    rel = diff / (abs(numeric_local) + 1e-12)
                    st.success(f"Difference: {diff:.6g} (relative: {rel:.6g})")

                    # auto-correct threshold (2% relative)
                    AUTO_CORRECT_REL_THRESHOLD = 0.02
                    if rel > AUTO_CORRECT_REL_THRESHOLD:
                        st.warning("Discrepancy detected — silently asking model to recompute full solution.")
                        corrected_text, corrected_final, correction_meta = request_correction_and_replace(
                            client=client,
                            orig_prompt=prompt,
                            model_text=model_text,
                            final_number=final_number,
                            local_numeric=numeric_local,
                            temperature=0.0,
                            max_tokens=max_tokens
                        )

                        # append corrected answer to conversation and replace displayed model_text
                        st.session_state.conv.append("assistant", corrected_text)
                        model_text = corrected_text
                        final_number = corrected_final

                        st.markdown("**Corrected model reply (silent):**")
                        st.markdown(f"<div class='card'><pre>{model_text}</pre></div>", unsafe_allow_html=True)
                        st.write("Corrected model final numeric ->", final_number)

                        # recompute diff/rel after correction (if possible)
                        if final_number is not None:
                            diff2 = abs(final_number - numeric_local)
                            rel2 = diff2 / (abs(numeric_local) + 1e-12)
                            st.success(f"After correction — Difference: {diff2:.6g} (relative: {rel2:.6g})")
                            if rel2 < 1e-6:
                                st.balloons()
                                st.info("Corrected model answer matches local evaluation ✅")
                            else:
                                st.error("Corrected model still diverges — please review steps manually.")
                        else:
                            st.error("Correction did not produce a detectable final numeric answer.")
                    else:
                        # within threshold
                        if rel < 1e-6:
                            st.balloons()
                            st.info("Model answer matches local evaluation closely ✅")
                        else:
                            st.info("Model and local evaluation are acceptably close (within threshold).")
                else:
                    st.info("No local numeric evaluation available to compare.")

            # Save metadata
            st.session_state.last_run_meta = {"model_meta": meta, "time": now_iso()}

with right:
    st.subheader("Conversation History & Actions")
    if st.session_state.conv.messages:
        for m in st.session_state.conv.messages[::-1]:
            role = m.role
            ts = m.timestamp
            if role == "user":
                st.markdown(f"**You** · <span class='muted'>{ts}</span>", unsafe_allow_html=True)
                st.write(m.text)
            else:
                st.markdown(f"**Model** · <span class='muted'>{ts}</span>", unsafe_allow_html=True)
                st.write(m.text)
    else:
        st.info("No conversation yet. Enter a math problem and click Solve.")

    st.markdown("---")
    # Export conversation
    if st.button("Export conversation JSON"):
        data = {"conversation": st.session_state.conv.as_list(), "meta": st.session_state.last_run_meta}
        st.download_button("Download JSON", data=json.dumps(data, indent=2), file_name="math_conversation.json", mime="application/json")

    if st.button("Export transcript (plain text)"):
        items = []
        for m in st.session_state.conv.messages:
            items.append(f"[{m.timestamp}] {m.role.upper()}:\n{m.text}\n")
        blob = "\n".join(items)
        st.download_button("Download TXT", data=blob, file_name="math_transcript.txt", mime="text/plain")

# Footer / Suggestions
st.markdown("---")
st.markdown(
    """
    **Enterprise suggestions**
    - Store API keys in a secure secrets manager (HashiCorp Vault / AWS Secrets Manager) in production.
    - Use a proper authentication layer (OAuth2 / SSO) before exposing the UI.
    - Add request/response logging to a centralized observability system (ELK/Datadog).
    - Add unit/integration tests for parsing & evaluation.
    - Add model response canonicalization and schema validation.
    - Add usage quotas and rate-limit handling (important for large models).
    """
)
