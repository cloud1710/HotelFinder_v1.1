# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(
    page_title="Giới thiệu",
    page_icon="ℹ️",
    layout="wide"
)

def inject_css():
    st.markdown("""
    <style>
    :root {
      --c-bg: linear-gradient(140deg,#e8f4f9,#ffffff 55%,#f3fbff);
      --c-accent: #0d5f89;
      --c-accent-grad: linear-gradient(95deg,#0d5f89,#1188bc 55%,#14a6d9);
      --c-border: #d4e5ee;
      --radius-xl: 28px;
      --radius-md: 16px;
      --transition: .35s cubic-bezier(.4,.2,.2,1);
    }
    body, .stApp {
      background: var(--c-bg) !important;
    }
    .block-container {
      padding-top: 1.4rem;
      max-width: 1180px;
    }
    .hero-wrapper {
      position:relative;
      background:linear-gradient(125deg,#ffffff,#f2f9fc 70%);
      border:1px solid var(--c-border);
      border-radius: var(--radius-xl);
      padding:48px 56px 46px;
      box-shadow:0 18px 50px -22px rgba(10,69,104,.25),0 8px 28px -12px rgba(10,69,104,.15);
      overflow:hidden;
    }
    .hero-wrapper:before, .hero-wrapper:after {
      content:"";
      position:absolute;
      width:460px;
      height:460px;
      border-radius:50%;
      background:radial-gradient(circle at 35% 35%,rgba(17,140,189,.20),rgba(17,140,189,0));
      top:-160px;
      right:-120px;
      filter:blur(4px);
      opacity:.75;
      pointer-events:none;
    }
    .hero-wrapper:after {
      width:320px;height:320px;
      top:auto;bottom:-120px;right:auto;left:-80px;
      background:radial-gradient(circle at 55% 55%,rgba(0,86,132,.18),rgba(0,86,132,0));
    }
    .mini-badge {
      display:inline-block;
      background:linear-gradient(90deg,#0d5f89,#1394c1);
      color:#ffffff;
      font-size:11px;
      font-weight:600;
      padding:6px 16px 7px;
      letter-spacing:1px;
      border-radius:100px;
      box-shadow:0 4px 14px -6px rgba(0,55,85,.55);
      text-transform:uppercase;
    }
    .hero-title {
      font-size:42px;
      font-weight:780;
      line-height:1.1;
      margin:18px 0 14px;
      background:linear-gradient(90deg,#0f5172,#1193c5);
      -webkit-background-clip:text;
      color:transparent;
      letter-spacing:.6px;
    }
    .hero-sub {
      font-size:16.8px;
      max-width:760px;
      color:#2b5870;
      font-weight:500;
      line-height:1.55;
    }
    @media (max-width:760px){
      .hero-wrapper { padding:38px 30px 40px; }
      .hero-title { font-size:34px; }
    }
    .section-title {
      margin:50px 0 18px;
      font-size:23px;
      font-weight:720;
      letter-spacing:.5px;
      background:linear-gradient(90deg,#0d587c,#13a4d6 70%,#1bc4ff);
      -webkit-background-clip:text;
      color:transparent;
    }
    .grid {
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(270px,1fr));
      gap:28px;
      margin-top:8px;
    }
    .info-card {
      position:relative;
      background:linear-gradient(150deg,#0e4b69,#0f5d7f 55%,#117ba3);
      border:1px solid rgba(255,255,255,.16);
      border-radius:24px;
      padding:22px 22px 20px;
      color:#ffffff;
      box-shadow:0 12px 32px -14px rgba(0,0,0,.55),0 6px 18px -10px rgba(8,72,104,.55),0 0 0 1px rgba(255,255,255,.06) inset;
      overflow:hidden;
      display:flex;
      flex-direction:column;
      gap:14px;
      min-height:230px;
      transition:var(--transition);
    }
    .info-card:before {
      content:"";
      position:absolute;
      inset:0;
      background:
        radial-gradient(at 18% 16%,rgba(255,255,255,.25),rgba(255,255,255,0) 60%),
        radial-gradient(at 82% 86%,rgba(255,255,255,.18),rgba(255,255,255,0) 62%);
      mix-blend-mode:overlay;
      opacity:.6;
      pointer-events:none;
      transition:.45s;
    }
    .info-card:hover {
      transform:translateY(-8px) scale(1.018);
      box-shadow:0 22px 56px -20px rgba(0,0,0,.70),0 12px 32px -14px rgba(6,84,116,.60),0 0 0 1px #1795c4 inset;
      border-color:#1795c4;
    }
    .info-card:hover:before { opacity:.82; }
    .card-icon {
      width:54px;height:54px;
      display:flex;align-items:center;justify-content:center;
      background:linear-gradient(135deg,rgba(255,255,255,.18),rgba(255,255,255,.05));
      border:1px solid rgba(255,255,255,.28);
      border-radius:18px;
      font-size:24px;
      box-shadow:0 4px 12px -4px rgba(0,0,0,.55),0 0 0 1px rgba(255,255,255,.18) inset;
      text-shadow:0 2px 4px rgba(0,0,0,.45);
    }
    .card-title {
      font-size:17px;
      font-weight:700;
      letter-spacing:.4px;
      text-shadow:0 2px 4px rgba(0,0,0,.4);
    }
    .card-text {
      font-size:14.3px;
      line-height:1.55;
      font-weight:500;
      color:#d1e9f4;
      text-shadow:0 1px 2px rgba(0,0,0,.45);
      flex-grow:1;
    }
    .member-grid {
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
      gap:26px;
      margin-top:8px;
    }
    .member-card {
      background:#ffffff;
      border:1px solid #d8e6ed;
      border-radius:24px;
      padding:22px 22px 20px;
      position:relative;
      overflow:hidden;
      display:flex;
      gap:18px;
      box-shadow:0 14px 36px -18px rgba(16,86,118,.28),0 6px 18px -10px rgba(16,86,118,.18);
      transition:var(--transition);
    }
    .member-card:before {
      content:"";
      position:absolute;
      inset:0;
      background:
        radial-gradient(at 18% 16%,rgba(17,132,175,.14),rgba(17,132,175,0) 60%),
        radial-gradient(at 82% 86%,rgba(17,132,175,.12),rgba(17,132,175,0) 66%);
      opacity:.65;
      pointer-events:none;
      mix-blend-mode:overlay;
      transition:.45s;
    }
    .member-card:hover {
      transform:translateY(-6px);
      box-shadow:0 22px 54px -22px rgba(0,0,0,.45),0 12px 32px -14px rgba(16,86,118,.40);
      border-color:#18a3d2;
    }
    .member-card:hover:before { opacity:.9; }
    .avatar {
      width:72px;height:72px;
      border-radius:20px;
      background:linear-gradient(145deg,#0d5f89,#13a2d2);
      display:flex;align-items:center;justify-content:center;
      color:#ffffff;font-size:30px;font-weight:700;
      letter-spacing:1px;
      box-shadow:0 6px 18px -8px rgba(0,0,0,.55),0 0 0 1px rgba(255,255,255,.25) inset;
      text-shadow:0 2px 4px rgba(0,0,0,.45);
      flex-shrink:0;
    }
    .m-name {
      font-size:17px;
      font-weight:700;
      color:#0f5071;
      letter-spacing:.35px;
    }
    .m-role {
      font-size:12px;
      font-weight:600;
      letter-spacing:1px;
      text-transform:uppercase;
      color:#1377a2;
      margin-top:2px;
    }
    .m-desc {
      font-size:13.4px;
      line-height:1.55;
      font-weight:500;
      color:#355c70;
      margin-top:6px;
    }
    .footer-note {
      margin:60px 0 20px;
      text-align:center;
      font-size:13px;
      color:#557a8d;
    }
    .soft-sep {
      height:1px;
      background:linear-gradient(90deg,rgba(0,95,140,.0),rgba(0,95,140,.35),rgba(0,95,140,.0));
      margin:56px 0 34px;
      opacity:.55;
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# HERO
with st.container():
    st.markdown('<div class="hero-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="mini-badge">ABOUT THE PROJECT</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Đồ án tốt nghiệp – Hệ thống Tìm kiếm & Gợi ý Khách sạn</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-sub">
          Nền tảng hỗ trợ người dùng tìm kiếm khách sạn dựa trên từ khóa tự nhiên, 
          tiêu chí lọc linh hoạt và gợi ý thông minh. Ứng dụng hướng tới việc kết hợp 
          tìm kiếm ngữ nghĩa (semantic search), đánh giá trải nghiệm thực tế và khả năng 
          mở rộng cho nhiều nguồn dữ liệu khác nhau.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# THÔNG TIN CHÍNH
st.markdown('<div class="section-title">Thông tin chung</div>', unsafe_allow_html=True)
info_cols = st.columns(3)
with info_cols[0]:
    st.markdown("""
    <div class="info-card">
      <div class="card-icon">👩‍🏫</div>
      <div class="card-title">Giảng viên hướng dẫn</div>
      <div class="card-text">
        ThS. Khuất Thuỳ Phương – định hướng phương pháp, bảo đảm chuẩn học thuật và hỗ trợ chiến lược triển khai.
      </div>
    </div>
    """, unsafe_allow_html=True)
with info_cols[1]:
    st.markdown("""
    <div class="info-card">
      <div class="card-icon">🎓</div>
      <div class="card-title">Tính chất đồ án</div>
      <div class="card-text">
        Một sản phẩm đồ án tốt nghiệp của nhóm J, tập trung vào xử lý ngôn ngữ tự nhiên, tìm kiếm vector và trải nghiệm người dùng.
      </div>
    </div>
    """, unsafe_allow_html=True)
with info_cols[2]:
    st.markdown("""
    <div class="info-card">
      <div class="card-icon">🧩</div>
      <div class="card-title">Mục tiêu</div>
      <div class="card-text">
        Xây dựng nền tảng tìm kiếm khách sạn nhanh, chính xác, mở rộng được; là bước đệm tích hợp thêm phân tích cảm xúc & gợi ý nâng cao.
      </div>
    </div>
    """, unsafe_allow_html=True)

# THÀNH VIÊN
st.markdown('<div class="section-title">Thành viên thực hiện</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style="font-size:14.4px;color:#365e73;font-weight:500;max-width:760px;line-height:1.55;margin-top:-6px">
      Các thành viên phụ trách nhiều mảng: thu thập & chuẩn hóa dữ liệu, thiết kế mô hình xử lý truy vấn, 
      xây dựng giao diện và tối ưu trải nghiệm tương tác.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="member-grid">', unsafe_allow_html=True)
# Member 1
st.markdown("""
<div class="member-card">
  <div class="avatar">T</div>
  <div>
    <div class="m-name">Phạm Đông Đức Tiến</div>
    <div class="m-role">Học viên 1</div>
    <div class="m-desc">
      Phụ trách xử lý dữ liệu mô tả khách sạn, xây dựng pipeline
      embedding & tối ưu truy vấn tìm kiếm kết hợp filter động.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Member 2
st.markdown("""
<div class="member-card">
  <div class="avatar">V</div>
  <div>
    <div class="m-name">Nguyễn Hoàng Vinh</div>
    <div class="m-role">Học viên 2</div>
    <div class="m-desc">
      Thiết kế giao diện, luồng tương tác, tích hợp semantic search,
      xây dựng phần gợi ý & điều hướng trang đa chức năng.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# PHẠM VI & HƯỚNG PHÁT TRIỂN
st.markdown('<div class="soft-sep"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Phạm vi & Hướng phát triển</div>', unsafe_allow_html=True)
st.markdown(
    """
    - Tích hợp thêm phân tích cảm xúc từ đánh giá người dùng để xếp hạng đa chiều.
    - Bổ sung mô hình reranker để cải thiện độ chính xác top-k kết quả.
    - Hỗ trợ gợi ý hội thoại (conversational refinement) khi người dùng đặt câu tự nhiên.
    - Mở rộng sang so sánh nhiều khách sạn song song.
    - Tối ưu hiệu năng với caching vector và batching truy vấn embedding.
    """)

# FOOTER NOTE
st.markdown('<div class="footer-note">© Nhóm J – Đồ án tốt nghiệp. Phiên bản trình diễn.</div>', unsafe_allow_html=True)