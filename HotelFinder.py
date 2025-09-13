# -*- coding: utf-8 -*-
"""
Home Page tìm kiếm khách sạn
Update: Gợi ý phổ biến (1 tên + 1 nút 'Khám phá')
Fix:
  - Sửa lỗi button 'Khám phá' không thực thi do để trong chuỗi HTML.
  - Sửa cú pháp HERO_TOP_IMAGE_RADIUS (20sim -> 20).
"""
import streamlit as st
from datetime import date, timedelta

TARGET_SEARCH_PAGE = "pages/4_Embeddings_Test.py"
SIMILAR_PAGE_PATH = "pages/5_Similar_Hotels.py"

# ============== CONFIG FLAGS ==============
DARK_INPUTS = False
EXTEND_HERO_GRADIENT = True
REMOVE_BODY_WHITE_GAP = False
UNDERLINE_STYLE = False

# ---- ẢNH HERO ----
HERO_IMAGE_URL = "https://cache.marriott.com/is/image/marriotts7prod/50514355-arts-hotel-april-2018-02:Pano-Hor?wid=1600&fit=constrain"
HERO_IMAGE_MODE = "top"           # "side" | "background" | "top"
HERO_IMAGE_SIDE_FLEX = (1.05, 0.95)
SHOW_HERO_IMAGE_ALWAYS = True
USE_PSEUDO = True
DEBUG_HERO_IMAGE_BORDER = False

# ---- Riêng cho mode "top" ----
HERO_TOP_IMAGE_USE_ASPECT = True
HERO_TOP_IMAGE_ASPECT = "16/6"
HERO_TOP_IMAGE_HEIGHT = 280
HERO_TOP_IMAGE_OVERLAY = True
HERO_TOP_IMAGE_RADIUS = 20     # <-- FIX ở đây (20sim gây SyntaxError)
HERO_TOP_IMAGE_ZOOM_HOVER = True

st.set_page_config(
    page_title="Tìm khách sạn - Home",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def safe_rerun():
    try:
        st.rerun()
    except Exception:
        pass


# ============== STYLE ==============
def inject_minimal_css():
    if DARK_INPUTS:
        vars_block = """
          --inp-bg: rgba(255,255,255,0.10);
          --inp-bg-hover: rgba(255,255,255,0.16);
          --inp-bg-focus: rgba(255,255,255,0.22);
          --inp-border: rgba(255,255,255,0.15);
          --inp-border-hover: rgba(255,255,255,0.25);
          --inp-border-focus: rgba(255,255,255,0.38);
          --inp-text: #ffffff;
          --inp-placeholder: #cfdde3;
          --label-color: #dfeef4;
          --tag-bg: linear-gradient(140deg,rgba(255,255,255,0.28),rgba(255,255,255,0.14));
          --tag-border: rgba(255,255,255,0.24);
          --slider-track: rgba(255,255,255,0.20);
          --slider-thumb-shadow: 0 0 0 5px rgba(255,255,255,0.30);
        """
    else:
        vars_block = """
          --inp-bg: #ffffff;
          --inp-bg-hover: #f5f7f9;
          --inp-bg-focus: #f0f5f9;
          --inp-border: #d5dde3;
          --inp-border-hover: #c2ccd4;
          --inp-border-focus: #4b94c7;
          --inp-text: #183642;
          --inp-placeholder: #6e8894;
          --label-color: #2d5668;
          --tag-bg: #edf4f8;
          --tag-border: #d2e2ec;
          --slider-track: #d9e4ec;
          --slider-thumb-shadow: #00000000;
        """

    if HERO_IMAGE_MODE == "background":
        wrapper_bg = f"""
          background:
            linear-gradient(125deg,rgba(14,49,70,0.82),rgba(23,91,125,0.78)),
            url('{HERO_IMAGE_URL}');
          background-size:cover;
          background-position:center;
          box-shadow:0 18px 52px -20px rgba(9,40,58,0.55);
        """
        title_color = "#ffffff"
        sub_color = "#e0edf3"
        badge_bg = "linear-gradient(90deg,#e9f3fb,#d6ecf9)"
        badge_color = "#17465c"
    else:
        wrapper_bg = "background:linear-gradient(135deg,#ffffff,#f5fafc);box-shadow:0 18px 50px -18px rgba(18,55,78,0.28);"
        title_color = "#123c52"
        sub_color = "#406377"
        badge_bg = "linear-gradient(90deg,#1d5b79,#237095)"
        badge_color = "#ffffff"

    img_debug_border = "outline:2px dashed #ff7a00;" if DEBUG_HERO_IMAGE_BORDER else ""
    hide_on_small = "" if SHOW_HERO_IMAGE_ALWAYS else "@media (max-width:1080px){ .hero-right {display:none;} }"

    pseudo_block = ""
    if HERO_IMAGE_MODE == "side" and USE_PSEUDO:
        pseudo_block = f"""
          .hero-right::before {{
            content:"";
            position:absolute;inset:0;
            background:
              linear-gradient(180deg,rgba(0,0,0,0.15),rgba(0,0,0,0.50)),
              url('{HERO_IMAGE_URL}');
            background-size:cover;
            background-position:center;
            filter:brightness(1.04);
            transition:.6s;
          }}
          .hero-right:hover::before {{
            transform:scale(1.04);
            filter:brightness(1.10);
          }}
        """

    if HERO_IMAGE_MODE == "top":
        if HERO_TOP_IMAGE_USE_ASPECT:
            top_image_block = f"""
              .hero-top-image {{
                aspect-ratio:{HERO_TOP_IMAGE_ASPECT};
                height:auto;
                min-height:180px;
              }}
            """
        else:
            top_image_block = f"""
              .hero-top-image {{
                height:{HERO_TOP_IMAGE_HEIGHT}px;
              }}
            """
    else:
        top_image_block = ""

    top_overlay_block = ""
    if HERO_IMAGE_MODE == "top" and HERO_TOP_IMAGE_OVERLAY:
        top_overlay_block = """
          .hero-top-image::after {
            content:"";
            position:absolute;
            inset:auto 0 0 0;
            height:46%;
            background:linear-gradient(180deg,rgba(0,0,0,0),rgba(0,0,0,0.55));
            pointer-events:none;
          }
        """

    top_zoom_block = ""
    if HERO_IMAGE_MODE == "top" and HERO_TOP_IMAGE_ZOOM_HOVER:
        top_zoom_block = """
          .hero-top-image img {
            transition:transform .85s cubic-bezier(.25,.7,.2,1), filter .6s;
          }
          .hero-top-image:hover img {
            transform:scale(1.05);
            filter:brightness(1.05);
          }
        """

    underline_block = ""
    if UNDERLINE_STYLE:
        underline_block = """
          .search-inline div[data-baseweb="input"],
          .search-inline div[data-baseweb="select"] > div,
          .search-inline div[data-baseweb="select"] > div:focus-within {
            border:0 !important;
            border-bottom:2px solid var(--inp-border-focus) !important;
            border-radius:0 !important;
            background:transparent !important;
            box-shadow:none !important;
          }
        """

    css = f"""
    <style>
    :root {{
      --transition:.22s cubic-bezier(.4,.2,.2,1);
      {vars_block}
    }}
    body {{
      font-family:"Inter",system-ui;
      {"background:linear-gradient(180deg,#0c1f2c 0%,#11384a 420px,#f5f8fa 420px);background-attachment:fixed;" if EXTEND_HERO_GRADIENT else ""}
    }}
    .stApp, .stApp > header, [data-testid="stHeader"] {{ background:transparent !important; }}
    .block-container {{
      padding-top:1.1rem;
      {"background:#ffffff !important;" if not REMOVE_BODY_WHITE_GAP else "background:transparent !important;"}
    }}
    .hero-wrapper {{
      position:relative;
      padding:{'40px 42px 38px' if HERO_IMAGE_MODE=='top' else '46px 48px 36px'};
      border-radius:22px;
      overflow:hidden;
      min-height:{'auto' if HERO_IMAGE_MODE=='top' else '360px'};
      display:flex;
      flex-direction:column;
      justify-content:center;
      gap:18px;
      {wrapper_bg}
    }}
    .hero-layout {{ width:100%; display:flex; gap:46px; align-items:stretch; }}
    .hero-layout-top {{ width:100%; display:flex; flex-direction:column; gap:26px; }}
    .hero-left {{
      flex:{HERO_IMAGE_SIDE_FLEX[0]};
      display:flex; flex-direction:column; gap:14px;
      position:relative; z-index:2; min-width:0;
    }}
    .hero-top-image {{
      position:relative;width:100%;border-radius:{HERO_TOP_IMAGE_RADIUS}px;
      overflow:hidden;background:#0e3248;{img_debug_border}display:block;line-height:0;
    }}
    .hero-top-image img {{ width:100%;height:100%;object-fit:cover;display:block; }}
    {top_image_block}
    {top_overlay_block}
    {top_zoom_block}
    .hero-right {{
      flex:{HERO_IMAGE_SIDE_FLEX[1]};position:relative;border-radius:18px;
      overflow:hidden;background:#092c40;display:flex;align-items:stretch;
      {img_debug_border}box-shadow:0 10px 28px -12px rgba(11,53,78,0.45);min-height:300px;
    }}
    .hero-right::after {{
      content:"";position:absolute;inset:auto 0 0 0;height:120px;
      background:linear-gradient(180deg,rgba(0,0,0,0),rgba(0,0,0,0.55));pointer-events:none;
    }}
    {pseudo_block}
    {hide_on_small}
    .hero-photo-credit {{
      position:absolute;right:10px;bottom:10px;font-size:11px;color:#fff;
      background:rgba(0,0,0,0.35);padding:4px 10px;border-radius:40px;
      backdrop-filter:blur(4px);letter-spacing:.5px;font-weight:500;z-index:5;
    }}
    .hero-title {{
      font-size:42px;line-height:1.12;margin:0 0 4px;
      color:{title_color};font-weight:700;letter-spacing:.4px;
    }}
    @media (max-width:760px){{
      .hero-title {{ font-size:34px; }}
    }}
    .hero-sub {{
      font-size:17px;font-weight:500;color:{sub_color};
      max-width:760px;opacity:.94;
    }}
    .small-badge {{
      background:{badge_bg};color:{badge_color};padding:4px 12px 5px;
      font-size:11px;font-weight:600;border-radius:32px;letter-spacing:1px;
      text-transform:uppercase;display:inline-block;
      box-shadow:{'0 4px 12px -4px rgba(0,0,0,0.25)' if HERO_IMAGE_MODE != 'background' else 'none'};
    }}
    .search-inline {{ position:relative;display:flex;flex-direction:column;gap:10px;margin-top:6px;z-index:3; }}
    .search-grid-row.search-grid-main {{
      display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-top:4px;align-items:start;
    }}
    @media (max-width:1150px){{ .search-grid-row.search-grid-main {{ grid-template-columns:repeat(3,1fr); }} }}
    @media (max-width:780px){{ .search-grid-row.search-grid-main {{ grid-template-columns:repeat(2,1fr); }} }}
    @media (max-width:520px){{ .search-grid-row.search-grid-main {{ grid-template-columns:1fr; }} }}
    .search-inline label {{
      color:var(--label-color)!important;font-size:12.4px;font-weight:600;
      letter-spacing:.45px;text-transform:uppercase;opacity:.88;
    }}
    .search-inline div[data-baseweb="input"],
    .search-inline div[data-baseweb="select"] > div {{
      background:var(--inp-bg)!important;border:1px solid var(--inp-border)!important;
      border-radius:12px!important;box-shadow:none!important;transition:var(--transition);
      padding:2px 8px!important;
    }}
    .search-inline div[data-baseweb="input"]:hover,
    .search-inline div[data-baseweb="select"] > div:hover {{
      background:var(--inp-bg-hover)!important;border-color:var(--inp-border-hover)!important;
    }}
    .search-inline div[data-baseweb="input"]:focus-within,
    .search-inline div[data-baseweb="select"] > div:focus-within {{
      background:var(--inp-bg-focus)!important;border-color:var(--inp-border-focus)!important;
      box-shadow:0 0 0 2px rgba(60,140,190,0.18)!important;
    }}
    .search-inline input[type="text"],
    .search-inline input[type="number"],
    .search-inline input[type="search"],
    .search-inline input[type="date"] {{
      background:transparent!important;color:var(--inp-text)!important;
      font-weight:500;-webkit-text-fill-color:var(--inp-text);padding:6px 4px!important;
    }}
    .search-inline input::placeholder {{ color:var(--inp-placeholder)!important;opacity:.75; }}
    .search-inline div[data-baseweb="tag"] {{
      background:var(--tag-bg)!important;border:1px solid var(--tag-border)!important;
      border-radius:10px!important;color:{'#fff' if DARK_INPUTS else '#1c465a'}!important;padding:2px 6px;
    }}
    .primary-inline-btn button {{
      background:linear-gradient(90deg,#1d5b79,#237095);border:0;color:#fff;
      font-weight:600;letter-spacing:.4px;padding:14px 26px;border-radius:14px;
      box-shadow:0 8px 22px -8px rgba(0,0,0,0.45);transition:.28s;
    }}
    .primary-inline-btn button:hover {{
      transform:translateY(-3px);box-shadow:0 14px 32px -10px rgba(0,0,0,0.55);filter:brightness(1.07);
    }}
    .reset-inline-btn button {{
      background:#eef4f7;border:1px solid #d3e1e9;color:#2c566c;
      font-weight:500;letter-spacing:.3px;padding:14px 18px;border-radius:14px;transition:.25s;
    }}
    .reset-inline-btn button:hover {{ background:#e4edf2; }}
    .inline-note {{
      font-size:12.5px;color:#4c6c7c;font-weight:500;margin-top:4px;
    }}
    .hero-wrapper, .section-title {{ animation:fadeIn .55s ease; }}
    @keyframes fadeIn {{
      from {{ opacity:0; transform:translateY(8px); }}
      to {{ opacity:1; transform:translateY(0); }}
    }}
    .section-title {{
      font-size:27px;font-weight:700;margin:28px 0 16px;letter-spacing:.6px;color:#11384a;
    }}
    .dest-card {{
      background:#ffffff;border-radius:20px;padding:18px 18px 16px;border:1px solid #e1e9ee;
      position:relative;overflow:hidden;box-shadow:0 6px 18px -6px rgba(0,0,0,0.10);
      transition:.28s cubic-bezier(.4,.2,.2,1);height:100%;display:flex;flex-direction:column;gap:10px;
    }}
    .dest-card:hover {{
      transform:translateY(-6px);box-shadow:0 14px 34px -8px rgba(0,0,0,0.18);border-color:#d2dee4;
    }}
    .dest-pic {{
      height:130px;border-radius:16px;background-size:cover;background-position:center;
    }}
    .explore-btn button {{
      all:unset;cursor:pointer;background:linear-gradient(90deg,#1d5b79,#236d8f);
      padding:8px 18px;border-radius:32px;font-size:12.6px;font-weight:600;
      color:#fff;letter-spacing:.4px;transition:.25s;
    }}
    .explore-btn button:hover {{
      filter:brightness(1.08);box-shadow:0 6px 18px -6px rgba(0,0,0,0.30);
      transform:translateY(-2px);
    }}
    .footer {{
      margin-top:50px;padding:34px 0 40px;font-size:13px;color:#5b6b74;
      border-top:1px solid #e1e9ef;background:transparent;
    }}
    {underline_block}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


inject_minimal_css()

# ============== HELPERS ==============
def switch_or_notice(page_path: str):
    if hasattr(st, "switch_page"):
        try:
            st.switch_page(page_path)
        except Exception as e:
            st.warning(f"Không chuyển được sang {page_path}: {e}")
    else:
        st.info(f"Streamlit chưa hỗ trợ switch_page. Mở thủ công: {page_path}")

def set_search_context(query: str, filters: dict):
    st.session_state["search_query"] = query
    st.session_state["search_filters"] = filters

def ensure_recent_list():
    if "recent_queries" not in st.session_state:
        st.session_state["recent_queries"] = []

def add_recent_query(q: str):
    ensure_recent_list()
    if q and q not in st.session_state["recent_queries"]:
        st.session_state["recent_queries"].insert(0, q)
        st.session_state["recent_queries"] = st.session_state["recent_queries"][:8]

today = date.today()

# ============== HERO ==============
with st.container():
    st.markdown('<div class="hero-wrapper">', unsafe_allow_html=True)

    if HERO_IMAGE_MODE == "top":
        st.markdown(
            f"""
            <div class="hero-top-image">
              <img src="{HERO_IMAGE_URL}" alt="Hero" loading="lazy" />
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="hero-left">', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="{"hero-layout" if HERO_IMAGE_MODE!="top" else "hero-layout-top"}">', unsafe_allow_html=True)
        st.markdown('<div class="hero-left">', unsafe_allow_html=True)

    st.markdown('<div class="small-badge">KHÁM PHÁ & ĐẶT PHÒNG NHANH</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Tìm khách sạn lý tưởng cho chuyến đi tiếp theo của bạn</h1>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">So sánh nhanh – Lọc thông minh – Gợi ý từ đánh giá thực tế. Bắt đầu ngay.</div>', unsafe_allow_html=True)

    st.markdown('<div class="search-inline">', unsafe_allow_html=True)
    user_query = st.text_input("Từ khóa", value=st.session_state.get("base_query", ""), key="main_query")

    st.markdown('<div class="search-grid-row search-grid-main">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    checkin = c1.date_input("Nhận phòng", value=today)
    checkout = c2.date_input("Trả phòng", value=today + timedelta(days=2))
    guests = c3.number_input("Khách", min_value=1, max_value=20, value=2, step=1)
    rooms = c4.number_input("Phòng", min_value=1, max_value=10, value=1, step=1)
    price_range = c5.slider("Khoảng giá (USD/đêm)", 0, 500, (40, 180), step=10)
    st.markdown('</div>', unsafe_allow_html=True)

    amenities_all = ["Hồ bơi", "Spa", "Gym", "Bãi biển riêng", "Đưa đón sân bay",
                     "Bữa sáng", "Chỗ đậu xe", "Nhà hàng", "Bar", "Thân thiện gia đình"]
    amenities = st.multiselect("Tiện ích", amenities_all, default=["Hồ bơi", "Bữa sáng"], key="amenities_ms")

    star_mode = "Bất kỳ"
    star_min, star_max = 0.0, 5.0
    with st.expander("⚙️ Tuỳ chọn nâng cao"):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        sort_pref = adv_col1.selectbox("Ưu tiên sắp xếp", ["Mặc định", "Điểm cao", "Giá thấp", "Gần trung tâm"])
        refundable = adv_col2.checkbox("Chỉ phòng hủy miễn phí", False)
        breakfast_only = adv_col3.checkbox("Chỉ gồm bữa sáng", False)
        st.markdown("---")
        star_mode = st.radio("Chọn sao", ["Bất kỳ", "Khoảng"], horizontal=True)
        if star_mode == "Khoảng":
            star_min, star_max = st.slider("Từ ... đến ...", 0.0, 5.0, (3.0, 5.0), step=0.5)
        st.markdown('<div class="inline-note">- Kết hợp từ khóa + sao + tiện ích để cải thiện độ liên quan. - Sau khi bấm Tìm hoặc chọn xu hướng, các trang Embeddings / Similar sẽ dùng lại bộ lọc này.</div>', unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns([1.1, 1])
    with btn_col1:
        st.markdown('<div class="primary-inline-btn">', unsafe_allow_html=True)
        search_pressed = st.button("🔍 Tìm", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with btn_col2:
        st.markdown('<div class="reset-inline-btn">', unsafe_allow_html=True)
        reset_click = st.button("↺ Reset", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    def current_filters_dict():
        return {
            "checkin": str(checkin),
            "checkout": str(checkout),
            "guests": guests,
            "rooms": rooms,
            "price_min": price_range[0],
            "price_max": price_range[1],
            "star_min": star_min,
            "star_max": star_max,
            "amenities": amenities,
            "sort_pref": locals().get("sort_pref", "Mặc định"),
            "refundable": locals().get("refundable", False),
            "breakfast_only": locals().get("breakfast_only", False)
        }

    if reset_click:
        for k in ["base_query", "search_query", "search_filters", "recent_queries"]:
            st.session_state.pop(k, None)
        safe_rerun()

    if search_pressed:
        query_effective = user_query.strip() or "khách sạn"
        filters = current_filters_dict()
        set_search_context(query_effective, filters)
        add_recent_query(query_effective)
        switch_or_notice(TARGET_SEARCH_PAGE)

    st.markdown('</div>', unsafe_allow_html=True)  # /search-inline
    st.markdown('</div>', unsafe_allow_html=True)  # /hero-left

    if HERO_IMAGE_MODE == "side":
        st.markdown('<div class="hero-right">', unsafe_allow_html=True)
        if not USE_PSEUDO:
            st.markdown(
                f'<img src="{HERO_IMAGE_URL}" alt="Hero" class="hero-img-el" loading="lazy" style="width:100%;height:100%;object-fit:cover;" />',
                unsafe_allow_html=True
            )
        st.markdown('<div class="hero-photo-credit">Ảnh minh hoạ</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if HERO_IMAGE_MODE != "top":
        st.markdown('</div>', unsafe_allow_html=True)  # /hero-layout

    st.markdown('</div>', unsafe_allow_html=True)  # /hero-wrapper

# ============== BELOW HERO ==============
if "recent_queries" in st.session_state and st.session_state["recent_queries"]:
    st.markdown("**Bạn đã tìm gần đây:**")
    for rq in st.session_state["recent_queries"]:
        if st.button(f"⟳ {rq}", key=f"rq_{rq}"):
            st.session_state["main_query"] = rq
            safe_rerun()

st.markdown("<div class='section-title'>Xu hướng tìm kiếm</div>", unsafe_allow_html=True)
trending_terms = ["gần biển", "gần trung tâm", "khu nghỉ dưỡng", "hồ bơi", "phòng gym"]
trend_cols = st.columns(len(trending_terms))
for i, term in enumerate(trending_terms):
    with trend_cols[i]:
        if st.button(term, key=f"trend_btn_{i}"):
            filters = {
                "checkin": str(date.today()),
                "checkout": str(date.today() + timedelta(days=2)),
                "guests": 2,
                "rooms": 1,
                "price_min": 40,
                "price_max": 180,
                "star_min": 0.0,
                "star_max": 5.0,
                "amenities": [],
                "sort_pref": "Mặc định",
                "refundable": False,
                "breakfast_only": False
            }
            set_search_context(term, filters)
            add_recent_query(term)
            switch_or_notice(TARGET_SEARCH_PAGE)

# --------- Gợi ý phổ biến ---------
st.markdown("<div class='section-title'>Gợi ý phổ biến</div>", unsafe_allow_html=True)
destinations = [
    {
        "name": "Khách Sạn Apollo Nha Trang",
        "tag": "Gần biển",
        "doc_id": 194,
        "img": "https://d41chssnpqdne.cloudfront.net/user_upload_by_module/chat_bot/files/80808203/iTkYNP9haE6q8aXq.jpg?Expires=1758897374&Signature=v5JJ2rpk9z9kl0NISj0wmNL1v6O8sVf~9TpNTSc~wNxB3t-~coV1ymyaXSZNSKFd4YSmrYESIksEF4vzxT-p3vij5ju5KQ5r5C3bFC58FpZREZKATtzW8dDItIWBNKAgqCTlQnzkfzi5arX2fOoc0w7m0n0czY65jUQH9cby4Q0KV6Q8v~C0FF62IcosSugSAVXlXt1Q5deD30qYMKRSLkSWiR1U11URdBhHvFF5LCctZJgBs-HXw~hHhT7lvloUOE7ccrDTBeNzrxxIqX9x9knwxgh2ukE4dZccb9qycPjzKqUQsgMuoNfq5M6HdBvFUrMcPMj9LFnMVjyWXUl8aw__&Key-Pair-Id=K3USGZIKWMDCSX",
        "desc": "Điểm dừng chân lý tưởng cho du khách muốn khám phá vẻ đẹp của thành phố biển nổi tiếng."
    },
    {
        "name": "Khách sạn AZURA",
        "tag": "Nghỉ dưỡng",
        "doc_id": 128,
        "img": "https://d41chssnpqdne.cloudfront.net/user_upload_by_module/chat_bot/files/80808203/GpNQ2nWrfkyg1BcN.jpg?Expires=1758897441&Signature=jBPzIUhX51e5wyCLZNqZbgWJRv6m-42YoaBrG9bmjiTvQJ8bJJ~3YUt0qTnHB7o3bqikWKnS77EzY9HbEE6WJebkNwWSLLS6tQK4gWsDJdzac9cnE3K~VNcktoleiHDyswarK3WbRoKt~Wbd1K8una0Q54~Ux44I1A3sxSq4~l7TwM5XTM7VwX13aW8rN6iYb4Pg1DqdElQabSOMnwX3eldNcvuwygQwuQkJNyz3mCxXGhWAq011lGyUiB302TGAIvUCTKf1RHGN4FdnXNjPCq5bSzgKCCFQr6C6LPkBk8hyFF7SpaTXwIXuLfn~LlNfDa1EN0yYe2Bp1rIVtPcgDg__&Key-Pair-Id=K3USGZIKWMDCSX",
        "desc": "Điểm dừng chân lý tưởng cho những du khách mong muốn tận hưởng kỳ nghỉ thư giãn và tiện nghi."
    },
    {
        "name": "Panorama Nha Trang Ocean View",
        "tag": "Trung tâm",
        "doc_id": 162,
        "img": "https://d41chssnpqdne.cloudfront.net/user_upload_by_module/chat_bot/files/80808203/IbcUYZfMrCxKISLS.jpg?Expires=1758897493&Signature=E6bZwysITFUneu52JS6Vmv92uL~gG~W521k5TbaUSDksI~EtgTzENdjZoEu~ln6NJeZ7fM1Fv9BX4nXXoCciEgk91ccIDL3NJw1sjXt18f3LFe8NZPx4BRM-KkT7YDlG9OIuRT1EPLjr7x-MRbTGRKoIP8gzxG48aYkV5Y63j~XUcxSIcphJ42jUg42uLVvxzDjYzRApUZV1iN-qsVjEl~w3LBLIlE4m2mQHnbyzuUiJtHafCIy2wuBmkFEyBZpDwCUu4ZpwgzFwIkx5LiOhShbinsoPo~1cUG-ivWiHu6u7~3Ne4o-HJT6pDV90AaOw2cSmkuxPhXrPdzFo8sOfhQ__&Key-Pair-Id=K3USGZIKWMDCSX",
        "desc": "Khách sạn này nằm ngay trung tâm thành phố, gần với tất cả địa điểm nổi tiếng."
    },
    {
        "name": "Senkotel Nha Trang",
        "tag": "Trung tâm",
        "doc_id": 136,
        "img": "https://d41chssnpqdne.cloudfront.net/user_upload_by_module/chat_bot/files/80808203/eUQmtlWef0B2E6wh.jpg?Expires=1758897561&Signature=X9NhVze4V~s6EuG9Ly~EsEPIPnXHK5Q5kHvLeGsT9rZnlwTAwXJyuwOydjK7sS85j8S8j7zQb2it6bNNnPYk0tc-rsAHkl-jDQoByDr2QKS0liZ1NFtWOGN~YelHmbdcxXPywPwzkDRUNeEdAGaAWmvaTsb4XdGatZnJ87er-fJMsVD3Gb~anioDLCXdZIYYzm0Zh2LIcMfcKlUaV7Pup4dHsXpMYMCQcwiVRs9EKz8XLWPEfBrUjgSDopnxlKDc58l6mxuLiV-d7mblh5C8Am8SPmZ1TCLV5B22F680xpwCvqKfzARBvk33olHAaf6v~cKDlrgHxD80UCEUZ35AJw__&Key-Pair-Id=K3USGZIKWMDCSX",
        "desc": "Với vị trí trung tâm, khách sạn chỉ cách trung tâm thành phố 0.01km, mang đến sự thuận tiện khi di chuyển và khám phá."
    }
]

cols_dest = st.columns(4)
for i, d in enumerate(destinations):
    with cols_dest[i]:
        # Khối HTML cho phần nội dung tĩnh (không chứa lệnh Python)
        st.markdown(
            f"""
            <div class="dest-card">
              <div class="dest-pic" style="background-image:url('{d['img']}');"></div>
              <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:6px;">
                <div style="flex:1;">
                  <div style="font-weight:600;font-size:15.8px;color:#15374b;letter-spacing:.25px;line-height:1.25;">{d['name']}</div>
                </div>
                <span style="background:#f0f7fb;color:#256286;font-size:11px;font-weight:600;padding:4px 10px;border-radius:30px;white-space:nowrap;">{d['tag']}</span>
              </div>
              <div style="font-size:13.3px;line-height:1.50;color:#4c5d66;font-weight:500;">{d['desc']}</div>
            """,
            unsafe_allow_html=True
        )
        # NÚT THỰC: tách ra ngoài chuỗi HTML
        with st.container():
            st.markdown('<div class="explore-btn">', unsafe_allow_html=True)
            if st.button("Khám phá", key=f"explore_{i}"):
                st.session_state["similar_doc_id"] = d["doc_id"]
                switch_or_notice(SIMILAR_PAGE_PATH)
            st.markdown('</div>', unsafe_allow_html=True)
        # Đóng thẻ card
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer">
      <div><strong>HotelFinder Platform</strong> · Giải pháp tìm & phân tích khách sạn thông minh.</div>
      <div style="margin-top:6px;">Liên hệ: support@example.com · © 2025 Hotel Intelligence Suite</div>
      <div style="margin-top:4px;font-size:11px;">Prototype UI · Popular Suggestions (single name + single CTA)</div>
    </div>
    """,
    unsafe_allow_html=True
)