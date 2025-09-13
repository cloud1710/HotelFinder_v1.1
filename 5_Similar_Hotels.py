# -*- coding: utf-8 -*-
"""
Trang 5: Khách sạn tương tự (TF-IDF) - Bản chỉnh sửa
Bổ sung:
  - Hiển thị đầy đủ khối "Khách sạn gốc" (mô tả, expander, metrics, Insight, nút Similar)
  - Label expander Insight đổi thành: "Muốn biết thêm chi tết?, Hãy click vào đây"
  - Nút '🔁 Tìm khách sạn tương tự' nằm dưới phần Insight (cả khách sạn gốc & khách sạn tương tự)
"""
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import streamlit as st
from gensim import corpora, models, similarities

# ================== CẤU HÌNH ==================
MODEL_DIR = Path("models/hotel_tfidf_v1")
HOTELS_CSV = "data/hotels_small.csv"
DEFAULT_TOP_K = 5
DECIMALS_STAR = 1
DECIMALS_SCORES = 1
DEFAULT_PREVIEW_CHARS = 600
DEFAULT_STAR_RANGE = (0.0, 5.0)

NAME_CANDIDATES = ["Hotel_Name", "Name", "Title"]
EN_NAME_CANDIDATES = ["Name_En", "Name_EN", "English_Name", "Hotel_Name_EN", "NameEnglish"]
COLUMN_CANDIDATES = {
    "description": ["Hotel_Description", "Description", "Desc", "Content", "Full_Text", "Text"],
    "score_location": ["Location", "Score_Location", "Location_Score"],
    "score_cleanliness": ["Cleanliness", "Clean", "Score_Cleanliness"],
    "score_facilities": ["Facilities", "Facility", "Score_Facilities"],
    "comment": ["Body", "Review", "Comment", "Review_Body", "Top_Comment"]
}
STAR_SYNONYMS = ["star", "stars", "hotel_star", "hotelstar", "star_rating", "rating_star", "rank"]

st.set_page_config(page_title="Khách sạn tương tự", page_icon="🛎️", layout="centered")

# ================== PATH TRANG 4 ==================
def get_search_page_path() -> str:
    try:
        if Path(__file__).parent.name.lower() == "pages":
            return "4_Embeddings_Test.py"
    except:
        pass
    return "pages/4_Embeddings_Test.py"

SEARCH_PAGE_PATH = get_search_page_path()

# ================== SAFE SWITCH PAGE ==================
def safe_switch_page(page_path: str):
    if hasattr(st, "switch_page"):
        try:
            st.switch_page(page_path)
        except Exception as e:
            st.error(f"Không switch được sang {page_path}: {e}")
    else:
        st.error("Streamlit chưa hỗ trợ switch_page.")
        st.info(f"Mở thủ công file: {page_path}")

def safe_rerun():
    try:
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            raise AttributeError
    except Exception:
        st.warning("Không rerun tự động được. Vui lòng reload (F5).")

# ================== CSS ==================
def ensure_page5_css():
    if "page5_css_done" in st.session_state:
        return
    st.markdown("""
    <style>
      .star-rating-wrap {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-family: "Segoe UI", Arial, sans-serif;
          line-height: 1;
      }
      .star-icons { font-size: 1.05rem; color: #f7b500; letter-spacing: 1px; }
      .star-icons .empty { color: #d0d0d0; }
      .star-rating-text { font-size: 0.85rem; color: #555; }
      .hotel-rating-block {
          line-height: 1.3;
          margin: 4px 0 10px 0;
          font-size: 0.95rem;
      }
      .hotel-rating-block div strong {
          font-weight: 600;
      }
      .hotel-comment {
          margin-top: 2px;
      }
      .base-hotel-badge {
          display:inline-block;
          background:linear-gradient(90deg,#1d6fa5,#2492c7);
          color:#fff;
          padding:4px 10px;
          font-size:11.5px;
          border-radius:30px;
          margin-left:8px;
          letter-spacing:.5px;
          font-weight:600;
      }
    </style>
    """, unsafe_allow_html=True)
    st.session_state["page5_css_done"] = True

# ================== LOAD TF-IDF ==================
@st.cache_resource
def load_tfidf_resources(model_dir: Path) -> Dict[str, Any]:
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Thiếu meta.json.")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    dictionary = corpora.Dictionary.load(str(model_dir / "dictionary.dict"))
    tfidf_model = models.TfidfModel.load(str(model_dir / "tfidf.model"))

    idx_path = model_dir / "similarity.index"
    index = None
    for cls in [similarities.SparseMatrixSimilarity,
                similarities.MatrixSimilarity,
                similarities.Similarity]:
        try:
            index = cls.load(str(idx_path))
            break
        except Exception:
            continue
    if index is None:
        raise RuntimeError("Không load được similarity.index.")

    hotel_ids = None
    order_pkl = model_dir / "hotel_id_order.pkl"
    json_map = model_dir / "doc_id_to_hotel_id.json"
    if order_pkl.exists():
        import joblib
        try:
            hotel_ids = joblib.load(order_pkl)
        except Exception:
            pass
    elif json_map.exists():
        with open(json_map, "r", encoding="utf-8") as f:
            id_map = json.load(f)
        hotel_ids = [id_map[str(i)] for i in range(len(id_map))]

    return {
        "meta": meta,
        "dictionary": dictionary,
        "tfidf_model": tfidf_model,
        "index": index,
        "hotel_ids": hotel_ids
    }

# ================== LOAD CSV ==================
@st.cache_data
def load_hotels(csv_path: str):
    import pandas as pd
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    for syn in STAR_SYNONYMS:
        if syn in lower_map:
            orig = lower_map[syn]
            if orig != "star":
                df = df.rename(columns={orig: "star"})
            break
    if "star" in df.columns:
        try:
            df["star"] = df["star"].astype(float)
        except:
            pass
    return df

# ================== CORPUS ==================
@st.cache_resource
def load_extended_corpus(model_dir: Path):
    tfidf_mm = model_dir / "tfidf_corpus.mm"
    bow_mm = model_dir / "corpus.mm"
    tfidf_corpus = None
    bow_corpus = None
    if tfidf_mm.exists():
        try:
            tfidf_corpus = corpora.MmCorpus(str(tfidf_mm))
        except:
            pass
    if bow_mm.exists():
        try:
            bow_corpus = corpora.MmCorpus(str(bow_mm))
        except:
            pass
    return tfidf_corpus, bow_corpus

def get_doc_vector(doc_id: int, tfidf_corpus, bow_corpus, tfidf_model):
    if tfidf_corpus is not None and 0 <= doc_id < len(tfidf_corpus):
        return tfidf_corpus[doc_id]
    if bow_corpus is not None and 0 <= doc_id < len(bow_corpus):
        return tfidf_model[bow_corpus[doc_id]]
    return None

def top_similar_from_index(base_doc_id: int,
                           index,
                           tfidf_corpus,
                           bow_corpus,
                           tfidf_model,
                           top_k: int) -> List[Tuple[int, float]]:
    vec = get_doc_vector(base_doc_id, tfidf_corpus, bow_corpus, tfidf_model)
    if vec is None:
        return []
    sims = np.array(index[vec])
    if base_doc_id < len(sims):
        sims[base_doc_id] = -1
    k = min(top_k, sims.size)
    if k <= 0:
        return []
    part = np.argpartition(-sims, k - 1)[:k]
    ordered = part[np.argsort(-sims[part])]
    return [(int(i), float(sims[i])) for i in ordered if sims[i] >= 0]

# ================== TEXT UTILS ==================
def pick_value(row, candidates: List[str]) -> str:
    for c in candidates:
        if c in row.index:
            val = row[c]
            if val is not None:
                sval = str(val).strip()
                if sval and sval.lower() not in ("nan", "none"):
                    return sval
    return "—"

def pick_english_name(row) -> str:
    return pick_value(row, EN_NAME_CANDIDATES)

def clean_description(raw: str) -> str:
    if raw is None:
        return "—"
    s = str(raw).strip()
    if not s:
        return "—"
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r'\s+', ' ', s)
    return s

SENT_SPLIT_REGEX = re.compile(r'(?<=[\.!?])\s+')

def split_sentences(text: str) -> List[str]:
    if not text or text == "—":
        return []
    parts = SENT_SPLIT_REGEX.split(text)
    return [p.strip() for p in parts if p.strip()]

def build_sentence_preview(text: str, max_chars: int) -> str:
    if text == "—":
        return text
    sents = split_sentences(text)
    if not sents:
        import textwrap
          # fallback shorten
        return text if len(text) <= max_chars else textwrap.shorten(text, width=max_chars, placeholder="…")
    out = []
    total = 0
    import textwrap
    for s in sents:
        add_len = len(s) + (2 if out else 0)
        if total + add_len > max_chars and out:
            break
        out.append(s)
        total += add_len
        if total >= max_chars:
            break
    preview = " ".join(out)
    if len(preview) > max_chars:
        preview = textwrap.shorten(preview, width=max_chars, placeholder="…")
    if len(preview) < len(text) and not preview.endswith(("…", ".", "!", "?")):
        preview += "..."
    return preview

def format_number(val, decimals=1) -> str:
    if val is None or val == "—":
        return "—"
    try:
        f = float(str(val).replace(",", "."))
    except:
        m = re.search(r'(\d+(?:[\.,]\d+)?)', str(val))
        if m:
            try:
                f = float(m.group(1).replace(",", "."))
            except:
                return "—"
        else:
            return "—"
    return f"{f:.{decimals}f}".rstrip("0").rstrip(".")

# ================== STAR RENDER ==================
def render_star_bar(rating: Optional[float], max_stars=5, decimals=DECIMALS_STAR) -> str:
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return '<span class="star-rating-text">Không rõ</span>'
    try:
        val = float(rating)
    except:
        return '<span class="star-rating-text">Không rõ</span>'
    rounded = int(round(val))
    rounded = min(max(rounded, 0), max_stars)
    num_str = f"{val:.{decimals}f}".rstrip("0").rstrip(".")
    filled = "★" * rounded
    empty = "☆" * (max_stars - rounded)
    stars_html = f'<span class="star-icons">{filled}<span class="empty">{empty}</span></span>'
    return f'<span class="star-rating-wrap">{stars_html}<span class="star-rating-text">{num_str} / {max_stars}</span></span>'

def extract_star_value(row) -> Optional[float]:
    if "star" not in row.index:
        return None
    try:
        v = float(row["star"])
        if np.isnan(v):
            return None
        return v
    except:
        return None

# ================== LỌC THEO SAO ==================
def filter_similars_by_star(similars: List[Tuple[int, float]],
                            hotels_df,
                            hotel_ids,
                            star_min: float,
                            star_max: float) -> List[Tuple[int, float]]:
    filtered = []
    for doc_id, score in similars:
        row, _ = get_row_for_doc(doc_id, hotels_df, hotel_ids)
        if row is None:
            continue
        sv = extract_star_value(row)
        if sv is None:
            continue
        if star_min <= sv <= star_max:
            filtered.append((doc_id, score))
    return filtered

def get_row_for_doc(doc_id: int, hotels_df, hotel_ids):
    if hotels_df is None:
        return None, None
    if hotel_ids is not None and doc_id < len(hotel_ids):
        hid = hotel_ids[doc_id]
        if "Hotel_ID" in hotels_df.columns:
            sub = hotels_df.loc[hotels_df["Hotel_ID"] == hid]
            if not sub.empty:
                return sub.iloc[0], hid
        if doc_id < len(hotels_df):
            return hotels_df.iloc[doc_id], hid
        return None, hid
    if doc_id < len(hotels_df):
        row = hotels_df.iloc[doc_id]
        hid = row["Hotel_ID"] if "Hotel_ID" in row.index else doc_id
        return row, hid
    return None, None

# ================== INSIGHT / Ý KIẾN KHÁCH HÀNG ==================
INSIGHT_POSSIBLE_METRICS = [
    "Location",
    "Cleanliness",
    "Facilities",
    "Service",
    "Value",
    "score",
    "value_for_money",
    "star"
]
METRIC_VI_MAP = {
    "Location": "Vị trí",
    "Cleanliness": "Sạch sẽ",
    "Facilities": "Cơ sở vật chất",
    "Service": "Dịch vụ",
    "Value": "Giá trị",
    "score": "Tổng điểm",
    "value_for_money": "Đáng tiền",
    "star": "Số sao"
}

VN_STOPWORDS = set("""
và hoặc nhưng thì là mà ở với về của cho từ đến tới những các cái rất rằng đã đang sẽ được
bị bởi như nữa thôi chỉ vẫn cũng này kia nọ đó đây ấy đâu nào vì nên nếu khi để tại trong ngoài
lên xuống hãy đừng chớ không chẳng chưa một hai ba bốn năm sáu bảy tám chín mười
""".split())

POS_WORDS = set("""
tuyệt tuyệt_vời tốt đẹp sạch thân_thiện nhiệt_tình thuận_tiện hài_lòng đáng_giá thoải_mái
xuất_sắc nhanh gọn ngon rộng rãi yên_tĩnh chuyên_nghiệp ấn_tượng hiện_đại an_toàn
""".split())

NEG_WORDS = set("""
tệ tồi bẩn ồn chậm nhỏ hẹp cũ xuống_cấp thất_vọng quá_tải đắt kém
khó_chịu phiền muộn lạnh_nhạt bất_tiện chật chội dơ hôi mốc hư hỏng
""".split())

TOKEN_COL_CANDIDATES = ["Content_done", "Content_no_stop_full_clean", "content_tokens"]
RAW_TEXT_CANDIDATES = ["Body", "Review", "Comment", "Review_Body", "Hotel_Description",
                       "Description", "Content", "Text"]

WORD_RE = re.compile(r"[0-9A-Za-zÀ-Ỵà-ỵ_]+")
try:
    from pyvi.ViTokenizer import tokenize as vi_tokenize
    HAS_PYVI = True
except Exception:
    HAS_PYVI = False

def simple_tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    t = text.lower()
    if HAS_PYVI:
        toks = vi_tokenize(t).split()
    else:
        toks = WORD_RE.findall(t)
    return [w for w in toks if len(w) > 1 and w not in VN_STOPWORDS]

def prepare_metrics_cache_vi(df):
    import pandas as pd
    if df is None or df.empty:
        return [], None, None
    metrics = [m for m in INSIGHT_POSSIBLE_METRICS if m in df.columns and m != "star"]
    if "Hotel_ID" not in df.columns or not metrics:
        return metrics, None, None
    per_hotel_mean = df.groupby("Hotel_ID")[metrics].mean(numeric_only=True)
    global_mean = per_hotel_mean.mean(numeric_only=True)
    return metrics, per_hotel_mean, global_mean

def compute_benchmarks_for_hotel_vi(hotel_id: str,
                                    metrics_list: List[str],
                                    per_hotel_mean,
                                    global_mean):
    import pandas as pd
    if per_hotel_mean is None or global_mean is None or hotel_id not in per_hotel_mean.index:
        return pd.DataFrame(columns=["metric","hotel_mean","system_mean","diff","percentile"])
    rows = []
    for m in metrics_list:
        h_val = per_hotel_mean.loc[hotel_id, m]
        if pd.isna(h_val) or pd.isna(global_mean[m]):
            continue
        diff = float(h_val - global_mean[m])
        ser = per_hotel_mean[m].dropna()
        pct = float(ser.rank(pct=True).get(hotel_id, np.nan))
        rows.append({
            "metric": m,
            "hotel_mean": round(float(h_val), 3),
            "system_mean": round(float(global_mean[m]), 3),
            "diff": round(diff, 3),
            "percentile": round(pct, 3) if not np.isnan(pct) else np.nan
        })
    if not rows:
        return pd.DataFrame(columns=["metric","hotel_mean","system_mean","diff","percentile"])
    import pandas as pd
    bench = (pd.DataFrame(rows)
             .sort_values("diff", ascending=False)
             .reset_index(drop=True))
    bench["metric_vi"] = bench["metric"].map(lambda x: METRIC_VI_MAP.get(x, x))
    return bench

def split_strengths_weaknesses_names_vi(bench, topn=3):
    if bench.empty:
        return [], []
    strengths_df = bench[bench["diff"] > 0].nlargest(topn, "diff")
    weaknesses_df = bench[bench["diff"] < 0].nsmallest(topn, "diff")
    strengths = strengths_df["metric_vi"].tolist()
    weaknesses = weaknesses_df["metric_vi"].tolist()
    if not weaknesses:
        tail = bench.nsmallest(topn, "diff")
        weaknesses = tail["metric_vi"].tolist()
    return strengths, weaknesses

def extract_tokens_for_hotel_vi(df_all, hotel_id: str) -> List[str]:
    if df_all is None or df_all.empty:
        return []
    if "Hotel_ID" in df_all.columns:
        sub = df_all[df_all["Hotel_ID"] == hotel_id]
    else:
        return []
    if sub.empty:
        return []
    for c in TOKEN_COL_CANDIDATES:
        if c in sub.columns:
            toks = []
            for s in sub[c].fillna("").astype(str):
                toks.extend([w for w in s.split() if w and w not in VN_STOPWORDS and len(w) > 1])
            return toks
    raw_parts = []
    for c in RAW_TEXT_CANDIDATES:
        if c in sub.columns:
            raw_parts.extend(sub[c].fillna("").astype(str).tolist())
    raw_text = " ".join(raw_parts)
    return simple_tokenize(raw_text)

def build_keyword_frames_vi(tokens: List[str], topk=30):
    import pandas as pd
    if not tokens:
        empty = pd.DataFrame(columns=["token","count"])
        return empty, empty, empty, 0, 0, 0
    from collections import Counter
    pos_tokens = [t for t in tokens if t in POS_WORDS]
    neg_tokens = [t for t in tokens if t in NEG_WORDS]
    def top_df(lst):
        if not lst:
            return pd.DataFrame(columns=["token","count"])
        c2 = Counter(lst)
        return (pd.DataFrame(c2.most_common(topk), columns=["token","count"])
                .sort_values("count", ascending=False))
    pos_df = top_df(pos_tokens)
    neg_df = top_df(neg_tokens)
    pos_count = len(pos_tokens)
    neg_count = len(neg_tokens)
    total_tokens = len(tokens)
    return None, pos_df, neg_df, pos_count, neg_count, total_tokens

def compute_positive_ratio(pos_count: int, neg_count: int):
    denom = pos_count + neg_count
    if denom == 0:
        return None
    return pos_count / denom

def render_benchmark_chart_vi(bench_df):
    if bench_df.empty:
        st.info("Không có dữ liệu để so sánh.")
        return
    try:
        import altair as alt
        dfp = bench_df.copy()
        domain_min = min(-0.01, dfp["diff"].min() * 1.1)
        domain_max = max(0.01, dfp["diff"].max() * 1.1)
        chart = (
            alt.Chart(dfp)
            .mark_bar()
            .encode(
                x=alt.X("diff:Q",
                        title="Chênh lệch so với trung bình",
                        scale=alt.Scale(domain=[domain_min, domain_max])),
                y=alt.Y("metric_vi:N",
                        sort=alt.Sort(field="diff", order="descending"),
                        axis=alt.Axis(title=None, labelLimit=130)),
                color=alt.condition("datum.diff > 0",
                                    alt.value("#27ae60"),
                                    alt.value("#e74c3c")),
                tooltip=[
                    alt.Tooltip("metric_vi:N", title="Chỉ số"),
                    alt.Tooltip("hotel_mean:Q", title="Điểm KS"),
                    alt.Tooltip("system_mean:Q", title="Trung bình"),
                    alt.Tooltip("diff:Q", title="Chênh lệch"),
                    alt.Tooltip("percentile:Q", title="Phân vị")
                ]
            )
        )
        rule = alt.Chart(dfp).mark_rule(color="#555", strokeDash=[4,4]).encode(x=alt.datum(0))
        st.altair_chart(chart + rule, use_container_width=True)
    except Exception as e:
        st.warning(f"Không vẽ được biểu đồ: {e}")
        st.dataframe(bench_df[["metric_vi","diff"]])

def render_positive_ratio_bar_vi(ratio, pos_count: int, neg_count: int):
    if ratio is None:
        pct = 50.0
    else:
        pct = ratio * 100
    html = f"""
    <div style="margin-top:6px;">
      <div style="
        position:relative;
        width:100%;
        height:40px;
        border-radius:14px;
        background: linear-gradient(90deg,#d64b4b 0%, #f2c94c 50%, #2ecc71 100%);
        box-shadow: inset 0 0 6px rgba(0,0,0,.35);
        overflow:hidden;
        font-family: 'Segoe UI', Arial, sans-serif;
      ">
        <div style="
          position:absolute;
          left:{pct}%;
          top:0;
          bottom:0;
          width:2px;
          background:#ffffff;
          box-shadow:0 0 4px rgba(0,0,0,0.55), 0 0 2px rgba(255,255,255,0.9);
        "></div>
        <div style="
          position:absolute;
          left:0;top:0;right:0;bottom:0;
          display:flex;
          align-items:center;
          justify-content:center;
          font-size:19px;
          font-weight:600;
          color:#111;
          text-shadow:0 1px 2px rgba(255,255,255,0.65);
        ">{pct:.1f}%</div>
        <div style="position:absolute;left:8px;top:6px;font-size:18px;">👎</div>
        <div style="position:absolute;right:8px;top:6px;font-size:18px;">👍</div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_insight_block_vi(hotel_id: str,
                            hotels_df,
                            cache_key: str = "insight_metrics_cache_vi_page5"):
    if cache_key not in st.session_state:
        metrics_list, per_hotel_mean, global_mean = prepare_metrics_cache_vi(hotels_df)
        st.session_state[cache_key] = {
            "metrics_list": metrics_list,
            "per_hotel_mean": per_hotel_mean,
            "global_mean": global_mean
        }
    cache = st.session_state[cache_key]
    metrics_list = cache["metrics_list"]
    per_hotel_mean = cache["per_hotel_mean"]
    global_mean = cache["global_mean"]

    with st.expander("Muốn biết thêm chi tết?, Hãy click vào đây"):
        st.markdown("### 📊 So sánh với các khách sạn khác")
        bench = compute_benchmarks_for_hotel_vi(hotel_id, metrics_list, per_hotel_mean, global_mean)
        strengths, weaknesses = split_strengths_weaknesses_names_vi(bench, topn=3)
        if bench.empty:
            st.write("Không đủ dữ liệu để so sánh.")
        else:
            render_benchmark_chart_vi(bench)
            col_s, col_w = st.columns(2)
            with col_s:
                st.markdown("**Điểm mạnh nổi bật**")
                if strengths:
                    st.markdown("- " + "\n- ".join(strengths))
                else:
                    st.markdown("_Chưa xác định_")
            with col_w:
                st.markdown("**Điểm yếu tương đối**")
                if weaknesses:
                    st.markdown("- " + "\n- ".join(weaknesses))
                else:
                    st.markdown("_Chưa xác định_")

        st.markdown("### 💬 Ý kiến khách hàng")
        _, pos_df, neg_df, pos_count, neg_count, total_tokens = build_keyword_frames_vi(
            extract_tokens_for_hotel_vi(hotels_df, hotel_id),
            topk=40
        )
        ratio = compute_positive_ratio(pos_count, neg_count)
        render_positive_ratio_bar_vi(ratio, pos_count, neg_count)

# ================== RENDER HOTEL BLOCK ==================
def render_hotel_block(rank: int,
                       doc_id: int,
                       row,
                       hid,
                       preview_char_limit: int,
                       show_comment: bool,
                       custom_title: Optional[str] = None):
    name_vn = pick_value(row, NAME_CANDIDATES)
    if name_vn == "—":
        name_vn = f"Khách sạn #{hid}"
    name_en = pick_english_name(row)
    if name_en == "—":
        name_en = ""
    if custom_title:
        title_line = custom_title
    else:
        title_line = f"{rank}. {name_vn}"
        if name_en and name_en.lower() != name_vn.lower():
            title_line += f" ({name_en})"

    raw_desc = pick_value(row, COLUMN_CANDIDATES["description"])
    desc_clean = clean_description(raw_desc)
    preview = build_sentence_preview(desc_clean, preview_char_limit)

    star_val = row["star"] if "star" in row.index else None
    star_html = render_star_bar(star_val)

    score_loc = format_number(pick_value(row, COLUMN_CANDIDATES["score_location"]), DECIMALS_SCORES)
    score_clean = format_number(pick_value(row, COLUMN_CANDIDATES["score_cleanliness"]), DECIMALS_SCORES)
    score_fac = format_number(pick_value(row, COLUMN_CANDIDATES["score_facilities"]), DECIMALS_SCORES)
    comment_raw = pick_value(row, COLUMN_CANDIDATES["comment"])
    comment = clean_description(comment_raw)

    st.markdown(f"### {title_line}")
    st.markdown(f"**Mô tả (tóm tắt):** {preview}")
    with st.expander("> Xem mô tả đầy đủ"):
        st.markdown(desc_clean)

    metrics_html = [
        f"<div><strong>Số sao:</strong> {star_html}</div>",
        f"<div><strong>Điểm vị trí:</strong> {score_loc}</div>",
        f"<div><strong>Sạch sẽ:</strong> {score_clean}</div>",
        f"<div><strong>Cơ sở vật chất:</strong> {score_fac}</div>"
    ]
    if show_comment and comment not in ['—']:
        metrics_html.append(f"<div class='hotel-comment'><strong>Comment:</strong> {comment}</div>")

    st.markdown(
        "<div class='hotel-rating-block'>" + "\n".join(metrics_html) + "</div>",
        unsafe_allow_html=True
    )

# ====== RENDER BASE HOTEL BLOCK (KHÁCH SẠN GỐC) ======
def render_base_hotel_block(doc_id: int,
                            row,
                            hid,
                            preview_char_limit: int,
                            show_comment: bool,
                            show_insight: bool,
                            hotels_df):
    base_title = f"{pick_value(row, NAME_CANDIDATES)}"
    render_hotel_block(
        rank=0,
        doc_id=doc_id,
        row=row,
        hid=hid,
        preview_char_limit=preview_char_limit,
        show_comment=show_comment,
        custom_title=base_title
    )
    if show_insight and hid is not None:
        render_insight_block_vi(str(hid), hotels_df)
    # Nút tìm tương tự lại (rerun)
    if st.button("🔁 Tìm khách sạn tương tự", key=f"btn_sim_after_insight_base_{doc_id}"):
        st.session_state["similar_doc_id"] = int(doc_id)
        safe_rerun()
    st.markdown("---")

# ================== BUTTON SAU INSIGHT (CHO SIMILARS) ==================
def similar_button_after_insight(doc_id: int):
    if st.button("🔁 Tìm khách sạn tương tự", key=f"btn_sim_after_insight_{doc_id}"):
        st.session_state["similar_doc_id"] = int(doc_id)
        safe_rerun()

# ================== PAGE SIMILAR ==================
def page_similar(res: Dict[str, Any],
                 hotels_df,
                 hotel_ids,
                 tfidf_corpus,
                 bow_corpus):
    ensure_page5_css()
    st.title("🛎️ Khách sạn tương tự")

    if "similar_doc_id" not in st.session_state:
        st.warning("Chưa có similar_doc_id. Quay lại trang tìm kiếm.")
        if st.button("⬅️ Quay lại trang tìm kiếm"):
            safe_switch_page(SEARCH_PAGE_PATH)
        return

    base_doc_id = int(st.session_state["similar_doc_id"])

    with st.sidebar:
        st.header("Tùy chọn")
        top_k = st.slider("Số gợi ý (top K)", 2, 20, DEFAULT_TOP_K, step=1)
        preview_char_limit = st.slider("Độ dài tóm tắt", 200, 1500, DEFAULT_PREVIEW_CHARS, step=50)
          # Áp dụng chung cho tất cả
        show_comment = st.checkbox("Hiển thị comment", True)
        show_insight = st.checkbox("Hiển thị Insight", True)
        enable_star_filter = st.checkbox("Bật lọc số sao", False)
        star_min, star_max = DEFAULT_STAR_RANGE
        if enable_star_filter:
            star_min, star_max = st.slider(
                "Khoảng sao", 0.0, 5.0, (1.0, 5.0),
                step=0.5, help="Chỉ hiển thị khách sạn có số sao trong khoảng này"
            )
        show_debug = st.checkbox("Debug", False)
        if st.button("⬅️ Quay lại tìm kiếm"):
            safe_switch_page(SEARCH_PAGE_PATH)

    index = res["index"]
    tfidf_model = res["tfidf_model"]

    row_base, hid_base = get_row_for_doc(base_doc_id, hotels_df, hotel_ids)
    if row_base is None:
        st.error("Không tìm thấy dữ liệu khách sạn gốc.")
        if st.button("Quay lại trang tìm kiếm"):
            safe_switch_page(SEARCH_PAGE_PATH)
        return

    # ----- HIỂN THỊ KHÁCH SẠN GỐC -----
    st.markdown("## 🔎 Khách sạn gốc được chọn")
    render_base_hotel_block(
        doc_id=base_doc_id,
        row=row_base,
        hid=hid_base,
        preview_char_limit=preview_char_limit,
        show_comment=show_comment,
        show_insight=show_insight,
        hotels_df=hotels_df
    )

    # ----- TÍNH SIMILAR -----
    similars = top_similar_from_index(
        base_doc_id,
        index,
        tfidf_corpus,
        bow_corpus,
        tfidf_model,
        top_k
    )

    if not similars:
        st.info("Không lấy được danh sách tương tự (cần tfidf_corpus.mm hoặc corpus.mm).")
        st.code("Huấn luyện cần lưu:\ncorpora.MmCorpus.serialize('tfidf_corpus.mm', tfidf_corpus)\n# hoặc\ncorpora.MmCorpus.serialize('corpus.mm', corpus_bow)")
        return

    original_count = len(similars)
    if enable_star_filter:
        similars = filter_similars_by_star(similars, hotels_df, hotel_ids, star_min, star_max)

    if enable_star_filter:
        st.markdown(f"*Áp dụng lọc sao: còn {len(similars)}/{original_count} kết quả nằm trong [{star_min}, {star_max}]*")

    if not similars:
        st.warning("Không còn khách sạn nào sau khi lọc số sao.")
        return

    st.markdown("## 🧩 Danh sách khách sạn tương tự")
    for rank, (doc_id, score) in enumerate(similars, 1):
        row, hid = get_row_for_doc(doc_id, hotels_df, hotel_ids)
        if row is None:
            st.markdown(f"### {rank}. (doc_id={doc_id}) Không tìm thấy dữ liệu.")
            st.markdown("---")
            continue

        render_hotel_block(
            rank=rank,
            doc_id=doc_id,
            row=row,
            hid=hid,
            preview_char_limit=preview_char_limit,
            show_comment=show_comment
        )

        if show_insight and hid is not None:
            render_insight_block_vi(str(hid), hotels_df)

        similar_button_after_insight(doc_id)

        st.markdown("---")

    if show_debug:
        with st.expander("Debug"):
            st.write("similar_doc_id:", base_doc_id)
            st.write("Base hotel_id:", hid_base)
            st.write("Similars (sau lọc):", similars)
            st.write("Original count:", original_count)
            st.write("Enable star filter:", enable_star_filter)
            st.write("Star range:", (star_min, star_max))
            st.write("Columns:", list(hotels_df.columns))
            st.write("SEARCH_PAGE_PATH:", SEARCH_PAGE_PATH)

    st.caption("Hoàn tất.")

# ================== MAIN ==================
def main():
    try:
        res = load_tfidf_resources(MODEL_DIR)
    except Exception as e:
        st.error(f"Lỗi tải mô hình: {e}")
        return

    hotels_df = load_hotels(HOTELS_CSV)
    if hotels_df is None:
        st.error(f"Không đọc được file: {HOTELS_CSV}")
        return

    tfidf_corpus, bow_corpus = load_extended_corpus(MODEL_DIR)
    page_similar(res, hotels_df, res["hotel_ids"], tfidf_corpus, bow_corpus)

if __name__ == "__main__":
    main()