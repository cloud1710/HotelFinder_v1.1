import streamlit as st
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from gensim import corpora, models, similarities

# ===============================
# CẤU HÌNH
# ===============================
MODEL_VERSION = "v1"
MODEL_DIR = Path(f"models/hotel_tfidf_{MODEL_VERSION}")

# ===============================
# TOKENIZER ĐƠN GIẢN
# (Có thể thay bằng underthesea, PyVi, vncorenlp...)
# ===============================
def tokenize(text: str) -> List[str]:
    return text.lower().split()

# ===============================
# HIGHLIGHT (Đơn giản, case-insensitive)
# ===============================
def highlight(content: str, tokens: List[str]) -> str:
    if not content:
        return content
    lowered = content.lower()
    spans = []
    for tk in tokens:
        if not tk.strip():
            continue
        start = 0
        while True:
            idx = lowered.find(tk, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(tk)))
            start = idx + len(tk)
    if not spans:
        return content
    spans.sort()
    merged = []
    cs, ce = spans[0]
    for s, e in spans[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    out = []
    last = 0
    for s, e in merged:
        out.append(content[last:s])
        out.append(f"<mark>{content[s:e]}</mark>")
        last = e
    out.append(content[last:])
    return "".join(out)

# ===============================
# LOAD TF-IDF (CACHE)
# ===============================
@st.cache_resource
def load_tfidf(model_dir: Path):
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Thiếu meta.json: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    dictionary = corpora.Dictionary.load(str(model_dir / "dictionary.dict"))
    tfidf_model = models.TfidfModel.load(str(model_dir / "tfidf.model"))

    # Thử load lần lượt nhiều loại index (tùy bạn build)
    idx_path = model_dir / "similarity.index"
    index = None
    load_errors = []
    for cls in [similarities.SparseMatrixSimilarity,
                similarities.MatrixSimilarity,
                similarities.Similarity]:
        try:
            index = cls.load(str(idx_path))
            break
        except Exception as e:
            load_errors.append((cls.__name__, str(e)))
    if index is None:
        raise RuntimeError(f"Không load được similarity.index. Tried: {load_errors}")

    doc_map_path = model_dir / "doc_id_to_text.json"
    if doc_map_path.exists():
        with open(doc_map_path, "r", encoding="utf-8") as f:
            doc_map = json.load(f)
    else:
        doc_map = None

    return {
        "meta": meta,
        "dictionary": dictionary,
        "tfidf_model": tfidf_model,
        "index": index,
        "doc_map": doc_map
    }

# ===============================
# SEARCH
# ===============================
def search_tfidf(query: str,
                 resources: Dict[str, Any],
                 topk: int = 5,
                 do_highlight: bool = True):
    dictionary = resources["dictionary"]
    tfidf_model = resources["tfidf_model"]
    index = resources["index"]
    doc_map = resources["doc_map"]

    tokens = tokenize(query)
    bow = dictionary.doc2bow(tokens)
    if not bow:
        return []

    vec = tfidf_model[bow]
    sims = index[vec]
    if isinstance(sims, list):
        sims = np.array(sims)

    n = len(sims)
    if n == 0:
        return []

    k = min(topk, n)
    # Lấy topK nhanh
    top_idx_part = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx_part[np.argsort(-sims[top_idx_part])]

    results = []
    for rank, doc_id in enumerate(top_idx, start=1):
        score = float(sims[doc_id])
        raw_text = doc_map.get(str(doc_id)) if doc_map else None
        if do_highlight and raw_text:
            h = highlight(raw_text, tokens)
        else:
            h = raw_text
        results.append({
            "rank": rank,
            "doc_id": int(doc_id),
            "score": score,
            "text": raw_text,
            "highlight": h
        })
    return results

# ===============================
# STREAMLIT UI DEMO
# ===============================
def run_app():
    st.title("Tìm kiếm TF-IDF (Khách sạn)")
    resources = load_tfidf(MODEL_DIR)
    meta = resources["meta"]

    st.sidebar.header("Thông tin model")
    st.sidebar.write(f"Version: {meta.get('model_version')}")
    st.sidebar.write(f"Số tài liệu: {meta.get('num_docs')}")
    st.sidebar.write(f"Số đặc trưng: {meta.get('num_features')}")

    query = st.text_input("Nhập truy vấn:", "")
    topk = st.slider("Top K", 1, 30, 5)
    highlight_opt = st.checkbox("Highlight", value=True)

    if query.strip():
        results = search_tfidf(query, resources, topk=topk, do_highlight=highlight_opt)
        if not results:
            st.warning("Không có kết quả (có thể toàn OOV).")
        else:
            for r in results:
                st.write(f"#{r['rank']} | doc_id={r['doc_id']} | score={r['score']:.4f}")
                if r["highlight"]:
                   st.markdown(r["highlight"], unsafe_allow_html=True)
                elif r["text"]:
                   st.write(r["text"])
                st.write("---")
    else:
        st.info("Nhập truy vấn để bắt đầu.")

if __name__ == "__main__":
    # Khi bạn chạy: streamlit run tfidf_app.py
    # Hàm run_app sẽ được gọi trong file chính hoặc bạn có thể bỏ if __name__...
    run_app()