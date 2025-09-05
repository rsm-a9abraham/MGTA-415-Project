# -*- coding: utf-8 -*-
# app2_allen.py — Retrieval-first Customer Intelligence Tool
# - Strict shopping intent
# - Gemini used for SMALL TALK + CLARIFYING QUESTIONS (no product suggestions)
# - No "sort by value"
# - No compare buttons; text prompt instead (typed compare still works)
# - Light-themed compare view for visibility

from __future__ import annotations

import os, re, ast, urllib.request
from typing import Optional, Tuple, List, Any
from html import escape

import numpy as np
import pandas as pd

import streamlit as st
from streamlit.components.v1 import html as st_html

st.set_page_config(page_title="📱 Customer Intelligence Tool", layout="wide")

# ====================== CONFIG ======================
USE_GEMINI_SMALLTALK = True     # Gemini for small talk
USE_GEMINI_CLARIFY = True       # Gemini for clarification prompts (no product naming)
SHOW_AI_JUDGE = True            # Simple "value score" judge card in compare view
RESULTS_TOP_N = 5               # Bullet list length

# ====================== STYLES ======================
st.markdown("""
<style>
.badge-row { display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 0 0; }
.badge-mini { font-size:12px; padding:6px 10px; border-radius:10px; border:1px solid transparent; }
.badge-mini.ok { background: #10b98122; border-color:#10b98155; color:#10b981; }
.badge-mini.err{ background: #ef444422; border-color:#ef444480; color:#ef4444; }

/* Comparison table (page-level styles; the compare view injects its own light CSS too) */
.cmp-table { width:100%; border-collapse:separate; border-spacing:0 10px; }
.cmp-row { background: rgba(148,163,184,0.10); border:1px solid rgba(148,163,184,0.38); }
.cmp-row td { padding:12px 14px; }
.cmp-row:hover { background: rgba(255,255,255,0.08); }
.cmp-key { width:180px; font-weight:700; }
.cmp-val { border-left:1px dashed rgba(148,163,184,0.45); }
.win-pill { display:inline-flex; align-items:center; gap:6px; padding:2px 10px; border-radius:999px;
            background:rgba(16,185,129,0.22); border:1px solid rgba(16,185,129,0.65); font-weight:700; color:#10b981; }
.win-pill::after { content:"✓"; font-size:12px; }

/* Typography */
.stChatMessage, .stMarkdown p, .stMarkdown li, .stMarkdown {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji" !important;
  font-size: 16px !important;
  line-height: 1.55 !important;
}

/* Example chips */
.suggest-row { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 0 0; }
.suggest { font-size:12px; padding:6px 10px; border-radius:999px; border:1px solid rgba(148,163,184,0.35);
           background:rgba(148,163,184,0.12); cursor: pointer; }
</style>
""", unsafe_allow_html=True)

# ====================== HEADER / RELOAD ======================
col_title, col_btn = st.columns([0.84, 0.16])
with col_title:
    st.title("💬 Customer Intelligence Tool (Cellphones & Accessories)")
    st.caption("Retrieval-based assistant — shopping answers come solely from the dataset; Gemini only for small talk & clarifying questions.")
with col_btn:
    def hard_reload():
        try:
            for k in list(st.session_state.keys()): del st.session_state[k]
        except Exception: pass
        try: st.cache_data.clear()
        except Exception: pass
        try: st.cache_resource.clear()
        except Exception: pass
        st.rerun()
    if st.button("🔁 Reload app", help="Fully clear caches and restart the app", use_container_width=True):
        hard_reload()

st.success("✅ App Working")

def badge(msg: str, ok: bool = True):
    c = "ok" if ok else "err"
    st.markdown(f"""<div class="badge-row"><div class="badge-mini {c}">{escape(msg)}</div></div>""",
                unsafe_allow_html=True)

# ====================== IDENTITY ======================
BOT_IDENTITY = (
    "You are the Customer Intelligence Tool for cellphones and accessories. "
    "For shopping: use only the curated dataset. "
    "For conversation: use Gemini for small talk and for asking clarifying questions; never suggest or name products/brands."
)
ASSISTANT_PROMPT = "What can I help you with today?"  # used only in welcome

# Example chips
QUICK_CHIPS_BASE = [
    "best iPhone 14 case under $20",
    "thin MagSafe case",
    "Pixel 7 screen protector",
]

# ====================== GEMINI (SMALL TALK + CLARIFY ONLY) ======================
GEMINI_OK = False
GEM_MODEL = "gemini-1.5-flash"

def load_api_key_from_file(path: str = "allen_apikey.txt") -> Optional[str]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                if s.upper().startswith("GOOGLE_API_KEY="):
                    return s.split("=", 1)[1].strip()
                return s
    except Exception:
        return None

if (USE_GEMINI_SMALLTALK or USE_GEMINI_CLARIFY):
    API_KEY = load_api_key_from_file()
    if API_KEY:
        os.environ["GOOGLE_API_KEY"] = API_KEY
    try:
        import google.generativeai as genai
        if API_KEY:
            genai.configure(api_key=API_KEY)
            GEMINI_OK = True
            badge("Gemini connected (small talk + clarify only)", ok=True)
        else:
            badge("Gemini key file not found (allen_apikey.txt). Clarify/small talk will use fallbacks.", ok=False)
    except Exception as e:
        badge(f"Gemini not active: {e}", ok=False)

def g_call(system_instruction: str, user_text: str) -> str:
    if not GEMINI_OK: return ""
    try:
        m = genai.GenerativeModel(GEM_MODEL, system_instruction=system_instruction)
        out = m.generate_content(user_text)
        txt = (out.text or "").strip()
        if not txt:
            out2 = m.generate_content(user_text + "\n\nPlease answer naturally in 1–2 short sentences.")
            txt = (out2.text or "").strip()
        return sanitize_md(txt) if txt else ""
    except Exception as e:
        badge(f"Gemini error: {e}", ok=False)
        return ""

def g_smalltalk(user_msg: str) -> str:
    if not USE_GEMINI_SMALLTALK: return "Hi! Ask me about a phone accessory when you’re ready."
    sys = (
        f"{BOT_IDENTITY} "
        "Task: small talk only. **Never** suggest, name, or imply any products or brands. "
        "Keep answers to 1–2 short sentences. No closing prompts."
    )
    txt = g_call(sys, user_msg)
    if txt: return txt
    # fallbacks
    m = user_msg.lower()
    if "name" in m or "what are you" in m or "what is this tool" in m:
        return "I’m the Customer Intelligence Tool for cellphones & accessories."
    if any(x in m for x in ["how are you", "how's it going", "how are u"]):
        return "I’m doing great—thanks!"
    if any(x in m for x in ["hello", "hi", "hey", "what's up"]):
        return "Hi there!"
    if any(x in m for x in ["thanks", "thank you", "thx"]):
        return "You’re welcome!"
    return "I’m here to help."

def g_clarify(user_msg: str) -> str:
    """Ask 1–2 targeted questions to disambiguate the user’s shopping intent. No product names."""
    if not USE_GEMINI_CLARIFY:
        return ("Could you share a bit more so I can search properly?\n"
                "- What phone model (e.g., iPhone 15 Pro, Galaxy S24, Pixel 7)?\n"
                "- What accessory (case, screen protector, charger, etc.) and any constraints (under $20, MagSafe, thin)?")
    sys = (
        f"{BOT_IDENTITY} "
        "Role: Ask 1–2 short, targeted clarification questions so you can search the catalog correctly. "
        "Focus on missing details like: phone model (e.g., iPhone 15 Pro / Galaxy S24 / Pixel 7), accessory type (case, screen protector, charger), "
        "key constraints (budget, MagSafe, material, slimness). "
        "**Do not** suggest or name any products or brands. "
        "Return only the questions, no preamble, no closing prompt."
    )
    txt = g_call(sys, user_msg)
    if txt: return txt
    # fallback
    return ("What phone model is this for (e.g., iPhone 15 Pro, Galaxy S24, Pixel 7)? "
            "Any constraints like budget, MagSafe, or material?")

# ====================== HEAVY DEPS (FAISS + MiniLM) ======================
HEAVY_OK = True
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim as _cos_sim
except Exception as e:
    HEAVY_OK = False
    badge(f"Failed to import FAISS / SentenceTransformer: {e}", ok=False)

# ====================== UTILS ======================
def price_str(v) -> str:
    try:
        if isinstance(v, (int, float, np.floating)) and not pd.isna(v):
            return f"${float(v):.2f}"
        if isinstance(v, str):
            m = re.search(r"[-+]?\d*\.?\d+", v)
            if m: return f"${float(m.group()):.2f}"
    except Exception: pass
    return "N/A"

def sanitize_md(s: str) -> str:
    if not s: return ""
    s = re.sub(r"</?[^>]+>", "", s)
    s = s.replace("`", "")
    s = re.sub(r"[_*~]+", "", s)
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\ufeff", "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s.strip()

def ensure_price_float(df: pd.DataFrame) -> pd.DataFrame:
    if "price_float" in df.columns: return df
    def _coerce(x):
        if isinstance(x, (int, float, np.floating)) and not pd.isna(x): return float(x)
        if isinstance(x, str):
            m = re.search(r"[-+]?\d*\.?\d+", x)
            if m:
                try: return float(m.group())
                except: return np.nan
        return np.nan
    df = df.copy()
    df["price_float"] = df.get("display_price", np.nan).map(_coerce)
    return df

def as_list(x):
    if isinstance(x, (list, tuple)): return list(x)
    if isinstance(x, (np.ndarray, pd.Series)):
        return [v for v in x.tolist() if pd.notna(v)]
    if isinstance(x, str) and x.strip():
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return [u for u in v if pd.notna(u)]
        except Exception: pass
        return [x]
    return []

# ====================== COMPARE RENDER ======================
def render_pretty_compare(left: dict, right: dict):
    def _pval(p):
        v = p.get("price_float") if "price_float" in p else p.get("display_price")
        try:
            if isinstance(v, (int,float,np.floating)) and not pd.isna(v): return float(v)
            if isinstance(v, str):
                m = re.search(r"[-+]?\d*\.?\d+", v)
                if m: return float(m.group())
        except Exception: pass
        return None

    def row(label, aval, bval, winner=None):
        acontent = f"<span class='win-pill'>{escape(str(aval))}</span>" if winner == "A" else escape(str(aval))
        bcontent = f"<span class='win-pill'>{escape(str(bval))}</span>" if winner == "B" else escape(str(bval))
        return f"""
        <tr class="cmp-row">
          <td class="cmp-key">{escape(str(label))}</td>
          <td class="cmp-val">{acontent}</td>
          <td class="cmp-val">{bcontent}</td>
        </tr>"""

    lp, rp = _pval(left), _pval(right)
    better_price = "A" if (lp is not None and rp is not None and lp < rp) else ("B" if (lp is not None and rp is not None and rp < lp) else None)
    lr, rr = left.get("average_rating"), right.get("average_rating")
    better_rate = "A" if (isinstance(lr,(int,float,np.floating)) and isinstance(rr,(int,float,np.floating)) and lr > rr) else ("B" if (isinstance(lr,(int,float,np.floating)) and isinstance(rr,(int,float,np.floating)) and rr > lr) else None)
    la, lb = left.get("rating_number"), right.get("rating_number")
    better_vol = "A" if (isinstance(la,(int,np.integer)) and isinstance(lb,(int,np.integer)) and la > lb) else ("B" if (isinstance(la,(int,np.integer)) and isinstance(lb,(int,np.integer)) and lb > la) else None)

    a_brand = left.get("store") or left.get("brand_clean") or "—"
    b_brand = right.get("store") or right.get("brand_clean") or "—"
    a_feat = ", ".join(map(str, as_list(left.get("features", []))[:8])) or "—"
    b_feat = ", ".join(map(str, as_list(right.get("features", []))[:8])) or "—"
    a_desc = (str(left.get("description", ""))[:400] or "—")
    b_desc = (str(right.get("description", ""))[:400] or "—")

    html = f"""
    <style>
      html, body {{
        background: #ffffff;
        color: #111827;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
        margin: 0; padding: 0;
      }}
      .cmp-table {{ width:100%; border-collapse:separate; border-spacing:0 10px; }}
      .cmp-row {{ background:#ffffff; border:1px solid rgba(0,0,0,0.12); border-radius:10px; }}
      .cmp-row:hover {{ background:#f9fafb; }}
      .cmp-row td {{ padding:12px 14px; vertical-align:top; }}
      .cmp-key {{ width:180px; color:#374151; font-weight:700; }}
      .cmp-val {{ border-left:1px dashed rgba(0,0,0,0.16); color:#111827; }}
      .win-pill {{
        display:inline-flex; align-items:center; gap:6px; padding:2px 10px; border-radius:999px;
        background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.45); font-weight:700; color:#047857;
      }}
      .win-pill::after {{ content:"✓"; font-size:12px; }}
    </style>
    <table class="cmp-table">
      {row("price", price_str(lp), price_str(rp), better_price)}
      {row("average_rating", left.get("average_rating","N/A"), right.get("average_rating","N/A"), better_rate)}
      {row("rating_number", la or 'N/A', lb or 'N/A', better_vol)}
      {row("brand/store", a_brand, b_brand, None)}
      {row("features", a_feat, b_feat, None)}
      {row("description", a_desc, b_desc, None)}
    </table>
    """
    st_html(html, height=420, scrolling=True)

# ====================== ENCODER / SHARDS ======================
HF_BASE = "https://huggingface.co/GovinKin/MGTA415database/resolve/main/"
PARQUETS = [
    "cellphones_clean_with_price000.parquet",
    "cellphones_clean_with_price-0001.parquet",
    "cellphones_clean_with_price-0002.parquet",
    "cellphones_clean_with_price-0003.parquet",
    "cellphones_clean_with_price-0004.parquet",
    "cellphones_clean_with_price-0005.parquet",
    "cellphones_clean_with_price-0006.parquet",
]
FAISS_FILES = [
    "cellphones_with_price000.faiss",
    "full-00001-of-00007.parquet.faiss",
    "full-00002-of-00007.parquet.faiss",
    "full-00003-of-00007.parquet.faiss",
    "full-00004-of-00007.parquet.faiss",
    "full-00005-of-00007.parquet.faiss",
    "full-00006-of-00007.parquet.faiss",
]

def _download_if_missing(remote_url: str, local_name: str):
    if os.path.exists(local_name): return local_name
    urllib.request.urlretrieve(remote_url + "?download=1", local_name)
    return local_name

if HEAVY_OK:
    @st.cache_resource(show_spinner=True)
    def load_encoder():
        return SentenceTransformer("all-MiniLM-L6-v2")
    try:
        model = load_encoder()
        badge("Encoder: all-MiniLM-L6-v2 ready", ok=True)
    except Exception as e:
        model = None
        HEAVY_OK = False
        badge(f"❌ Failed to load encoder: {e}", ok=False)
else:
    model = None

@st.cache_data(show_spinner=True)
def load_shards():
    dfs, idxs, total = [], [], 0
    for pq, fx in zip(PARQUETS, FAISS_FILES):
        pq_path = _download_if_missing(HF_BASE + pq, pq)
        fx_path = _download_if_missing(HF_BASE + fx, fx)
        df = pd.read_parquet(pq_path)
        idx = faiss.read_index(fx_path)
        dfs.append(df); idxs.append(idx); total += len(df)
    return dfs, idxs, total

dfs, idxs, total_rows = [], [], 0
try:
    if HEAVY_OK:
        dfs, idxs, total_rows = load_shards()
        badge(f"Loaded {len(dfs)} shard pairs from HF ({total_rows:,} rows total).", ok=True)
        badge(f"Dataset loaded: {total_rows:,} rows", ok=True)
    else:
        badge("❌ Skipped loading shards (FAISS/ST not imported).", ok=False)
except Exception as e:
    badge(f"❌ Failed to load dataset/index bundle: {e}", ok=False)

# ====================== RETRIEVAL ======================
def search_all_shards(query: str, model, dfs: List[pd.DataFrame], idxs: List[Any],
                      top_k_per=10, final_top=30) -> pd.DataFrame:
    if model is None or not dfs or not idxs: return pd.DataFrame()
    qv = model.encode([query]).astype("float32")
    rows = []
    for df, index in zip(dfs, idxs):
        try:
            D, I = index.search(qv, k=top_k_per)
            sub = df.iloc[I[0]].copy()
            sub["distance"] = D[0]
            rows.append(sub)
        except Exception:
            continue
    if not rows: return pd.DataFrame()
    merged = pd.concat(rows, ignore_index=True).sort_values("distance", ascending=True).head(final_top)
    return merged

def rerank_by_similarity(query: str, results: pd.DataFrame, model, top_n=RESULTS_TOP_N) -> pd.DataFrame:
    if results.empty or model is None: return results
    query_vec = model.encode([query], convert_to_tensor=True)
    titles = results["title"].astype(str).tolist()
    title_vecs = model.encode(titles, convert_to_tensor=True)
    sims = _cos_sim(query_vec, title_vecs)[0].cpu().numpy()
    out = results.copy()
    out["similarity_score"] = sims
    return out.sort_values("similarity_score", ascending=False).head(top_n)

# ====================== INTENT / ACTIONS ======================
RE_COMPARE = re.compile(r"\b(?:compare|vs|versus)\b", re.I)
RE_NUMPAIR = re.compile(r"\b([1-9]|10)\s*(?:&|and|,|\s)\s*([1-9]|10)\b")
RE_LETTERPAIR = re.compile(r"\b([A-Ea-e])\s*(?:&|and|,|\s)\s*([A-Ea-e])\b")
RE_PRICE_RANGE = re.compile(r"between\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
RE_UNDER = re.compile(r"(?:under|below|less than|cheaper than|<)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
RE_OVER  = re.compile(r"(?:over|above|greater than|higher than|>)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)

PRODUCT_KEYWORDS = [
    # categories
    "case","cases","screen protector","protector","charger","charging","battery","power bank",
    "magsafe","mag-safe","wallet","kickstand","earbuds","airpods","headphones","cable","usb-c","lightning",
    # devices / brands
    "iphone","ipad","samsung","galaxy","pixel","android","oneplus","xiaomi","huawei","motorola",
    # common models
    "iphone 11","iphone 12","iphone 13","iphone 14","iphone 15","pixel 6","pixel 7","pixel 8",
    "s22","s23","s24","note 20","a54",
]

SHOPPING_SIGNALS = [
    "best","recommend","suggest","option","pick","looking for","buy","budget",
    "under","below","cheap","value","rating","reviews","protect","thin","slim","leather","magsafe","mag-safe","kickstand"
]

DEVICE_REGEXES = [
    r"\biphone\s?(?:se|xr|xs|11|12|13|14|15)(?:\s?(?:mini|plus|pro(?:\smax)?|max))?\b",
    r"\bpixel\s?(?:[3-9]\w?|\d{2})(?:\s?(?:a|pro))?\b",
    r"\bgalaxy\s?(?:s|note|a)\s?\d{1,2}\b",
    r"\bs2[0-9]\b",
    r"\bnote\s?\d{1,2}\b",
    r"\ba\d{2}\b"
]

def contains_product_keyword(msg: str) -> bool:
    m = msg.lower()
    return any(k in m for k in PRODUCT_KEYWORDS)

def has_shopping_signals_only(msg: str) -> bool:
    m = msg.lower()
    return any(k in m for k in SHOPPING_SIGNALS) and not contains_product_keyword(m)

def mentions_device_model(msg: str) -> bool:
    t = msg.lower()
    return any(re.search(p, t) for p in DEVICE_REGEXES)

def is_device_specific_category(msg: str) -> bool:
    t = msg.lower()
    return any(x in t for x in ["case", "cases", "screen protector", "protector"])

def needs_clarification(msg: str) -> bool:
    t = msg.lower()
    device_specific = is_device_specific_category(t)
    has_device = mentions_device_model(t)
    has_signals_only = has_shopping_signals_only(t)
    if device_specific and not has_device: return True
    if has_signals_only: return True
    if ("case" in t or "screen protector" in t) and not has_device: return True
    return False

def is_action_only_query(msg: str) -> bool:
    m = msg.lower()
    if RE_PRICE_RANGE.search(m) or RE_UNDER.search(m) or RE_OVER.search(m): return True
    if "sort" in m and any(x in m for x in ["price","rating","review","reviews"]): return True
    return False

def parse_pair(text: str, top_len: int) -> Tuple[Optional[int], Optional[int]]:
    m = RE_NUMPAIR.search(text or "")
    if m:
        i, j = int(m.group(1))-1, int(m.group(2))-1
        if 0 <= i < top_len and 0 <= j < top_len and i != j: return i, j
    m = RE_LETTERPAIR.search(text or "")
    if m:
        mapL = {c:i for i,c in enumerate("ABCDE")}
        i, j = mapL[m.group(1).upper()], mapL[m.group(2).upper()]
        if 0 <= i < top_len and 0 <= j < top_len and i != j: return i, j
    return None, None

def apply_text_action(user_msg: str, df: pd.DataFrame):
    """Apply filter/sort actions inferred from text. 'Sort by value' is DISABLED."""
    msg = user_msg.lower()
    df = ensure_price_float(df)
    r = RE_PRICE_RANGE.search(msg)
    if r:
        lo, hi = float(r.group(1)), float(r.group(2))
        if lo > hi: lo, hi = hi, lo
        flt = df[(df["price_float"] >= lo) & (df["price_float"] <= hi)].copy()
        return flt, f"Filtered between ${lo:.0f} and ${hi:.0f}."
    r = RE_UNDER.search(msg)
    if r:
        cap = float(r.group(1))
        flt = df[df["price_float"] < cap].copy()
        return flt, f"Filtered under ${cap:.0f}."
    r = RE_OVER.search(msg)
    if r:
        lo = float(r.group(1))
        flt = df[df["price_float"] > lo].copy()
        return flt, f"Filtered over ${lo:.0f}."
    if "sort" in msg:
        if "price" in msg:
            asc = "low" in msg or "asc" in msg
            return df.sort_values("price_float", ascending=asc).copy(), f"Sorted by price ({'low→high' if asc else 'high→low'})."
        if "rating" in msg:
            return df.sort_values("average_rating", ascending=False).copy(), "Sorted by rating (highest→lowest)."
        if "review" in msg:
            return df.sort_values("rating_number", ascending=False).copy(), "Sorted by reviews (most→least)."
        if "value" in msg:
            return df.copy(), "Note: sorting by value is disabled."
    return df, None

# ====================== CHAT STATE ======================
if "chat" not in st.session_state: st.session_state.chat = []
if "last_query" not in st.session_state: st.session_state.last_query = ""
if "last_top5" not in st.session_state: st.session_state.last_top5 = []

def reset_conversation():
    st.session_state.chat = []
    st.session_state.last_query = ""
    st.session_state.last_top5 = []

# ====================== HISTORY ======================
for m in st.session_state.chat:
    role = m.get("role","assistant")
    with st.chat_message(role) if hasattr(st, "chat_message") else st.container():
        st.markdown(m.get("content",""))

# ====================== WELCOME ======================
if not st.session_state.chat:
    hello = (
        "Hi! I’m the **Customer Intelligence Tool** for **cellphones & accessories**. "
        "Ask naturally — e.g., **best iPhone 14 case under $20**, **thin MagSafe case**, **Pixel 7 screen protector**.\n\n"
        f"_{ASSISTANT_PROMPT}_"
    )
    st.session_state.chat.append({"role":"assistant","content":hello})
    with st.chat_message("assistant"): st.markdown(hello)

# ====================== EXAMPLE CHIPS ======================
st.markdown('<div class="suggest-row">', unsafe_allow_html=True)
chip_cols = st.columns(len(QUICK_CHIPS_BASE))
chip_clicked_text = None
for i, label in enumerate(QUICK_CHIPS_BASE):
    if chip_cols[i].button(label, key=f"ex_chip_{i}"):
        chip_clicked_text = label
st.markdown('</div>', unsafe_allow_html=True)

# ====================== ALWAYS SHOW CHAT INPUT ======================
typed_msg = st.chat_input("Type a message (e.g., “thin MagSafe case”, “best iPhone 14 case under $20”).")
user_msg = chip_clicked_text or typed_msg

# ====================== CORE ACTIONS ======================
def handle_compare_request(text: str) -> bool:
    i, j = parse_pair(text, len(st.session_state.last_top5))
    if i is None or j is None or not st.session_state.last_top5:
        return False
    left, right = st.session_state.last_top5[i], st.session_state.last_top5[j]
    bloc = f"Comparing **#{i+1}** and **#{j+1}** from your current list:"
    with st.chat_message("assistant"):
        st.markdown(bloc)
        render_pretty_compare(left, right)
        if SHOW_AI_JUDGE:
            try:
                from crew_judge_module import judge_products, render_streamlit_card
                result = judge_products(st.session_state.last_query or text, left, right, use_crewai=True, model="gemini-1.5-flash")
                render_streamlit_card(result)
            except Exception:
                # Fallback local judge
                def val(p):
                    try:
                        pr = float(p.get("price_float") or np.inf)
                        ra = float(p.get("average_rating") or 0.0)
                        return ra / pr if pr > 0 else 0.0
                    except Exception:
                        return 0.0
                a, b = val(left), val(right)
                st.success(f"**Local Verdict** → Winner: **{'A' if a>=b else 'B'}**")
                st.caption(f"Value scores — A:{a:.3f} vs B:{b:.3f}")
        st.markdown("\n**Anything else I can help you with?**")
    st.session_state.chat.append({"role":"assistant","content":bloc + "\n\n**Anything else I can help you with?**"})
    return True

def show_clarification(msg: str, reason: str = ""):
    prompt = g_clarify(msg)
    with st.chat_message("assistant"):
        st.markdown(prompt)
    st.session_state.chat.append({"role":"assistant","content":prompt})

def run_retrieval_and_reply(query: str):
    st.session_state.last_query = query
    with st.spinner("Searching catalog…"):
        res = search_all_shards(query, model, dfs, idxs, top_k_per=12, final_top=48)
    if res.empty:
        show_clarification(query, reason="no_results")
        return

    res = ensure_price_float(res)
    pool, summary = apply_text_action(query, res)
    if pool is None or pool.empty: pool = res

    top = rerank_by_similarity(query, pool, model, top_n=RESULTS_TOP_N)
    if len(top) < 3:
        extra = pool[~pool.index.isin(top.index)].head(3 - len(top))
        top = pd.concat([top, extra], axis=0)

    tmp = top.copy()
    tmp["price"] = tmp.get("display_price", "N/A")
    tmp["average_rating"] = tmp.get("average_rating", np.nan)
    tmp["rating_number"] = tmp.get("rating_number", np.nan)
    tmp["features"] = tmp.get("features", [])
    tmp["categories"] = tmp.get("categories", [])
    tmp["description"] = tmp.get("description", "")
    if "store" not in tmp.columns and "brand_clean" in tmp.columns:
        tmp["store"] = tmp["brand_clean"]
    st.session_state.last_top5 = tmp.to_dict(orient="records")

    # Build bulleted results
    bullets = []
    for i, r in enumerate(st.session_state.last_top5, 1):
        p = price_str(r.get("price_float") if "price_float" in r else r.get("display_price"))
        rt = r.get("average_rating"); rt_s = f"{float(rt):.2f}" if pd.notna(rt) else "N/A"
        rv = r.get("rating_number"); rv_s = f"{int(rv):,}" if pd.notna(rv) else "N/A"
        bullets.append(f"{i}. **{str(r.get('title',''))[:90]}** — {p}, ⭐ {rt_s} ({rv_s})")

    pre = "Here are good options I found in the catalog."
    if summary: pre += " " + summary  # e.g., “Filtered under $20.”
    tail = ("\n\n**Would you like to compare any two products?** "
            "Reply like: `compare 1 & 3` or `1 and 3`.")
    final = pre + "\n\n" + "\n\n".join(bullets) + tail

    with st.chat_message("assistant"):
        st.markdown(final)
    st.session_state.chat.append({"role":"assistant","content":final})

# ====================== MAIN ======================
try:
    if user_msg:
        st.session_state.chat.append({"role":"user","content":user_msg})
        with st.chat_message("user"): st.markdown(user_msg)

        # 1) Explicit compare typed by user
        if RE_COMPARE.search(user_msg):
            if handle_compare_request(user_msg):
                pass
            else:
                reply = "I can compare after we list some products. Try a product query first (e.g., *iPhone 14 case under $20*)."
                st.session_state.chat.append({"role":"assistant","content":reply})
                with st.chat_message("assistant"): st.markdown(reply)

        # 2) Action-only tweaks (sort/filter) applied to last query
        elif is_action_only_query(user_msg) and st.session_state.last_query:
            run_retrieval_and_reply(st.session_state.last_query + " " + user_msg)

        # 3) Clarify if the ask is underspecified
        elif needs_clarification(user_msg):
            show_clarification(user_msg, reason="underspecified")

        # 4) Full shopping ask (sufficiently specific)
        elif contains_product_keyword(user_msg):
            if HEAVY_OK and model is not None and dfs and idxs:
                run_retrieval_and_reply(user_msg)
            else:
                msg = "Retrieval components aren’t loaded. Please ensure FAISS and SentenceTransformers are installed."
                st.session_state.chat.append({"role":"assistant","content":msg})
                with st.chat_message("assistant"): st.markdown(msg)

        # 5) Everything else is small talk
        else:
            reply = g_smalltalk(user_msg)
            st.session_state.chat.append({"role":"assistant","content":reply})
            with st.chat_message("assistant"): st.markdown(reply)

except Exception as e:
    st.error(f"Something went wrong handling your last message: {e}")
    st.info("You can keep chatting or tap **Reload app** (top-right) to fully restart.")

