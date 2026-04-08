"""
Search Page — Hybrid Search (Vector + BM25 + Re-ranking).
"""

import streamlit as st
from pathlib import Path

# Load CSS
css_path = Path(__file__).parents[1] / "styles" / "custom.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Show custom sidebar
from modules.sidebar import show_sidebar
show_sidebar()

st.markdown("# 🔍 Keyword Search")
st.markdown("ค้นหาเอกสารด้วย Keyword")

st.markdown("---")

# ─── Search Input ────────────────────────────────────────
query = st.text_input(
    "🔎 ค้นหา",
    placeholder="พิมพ์คำค้นหา เช่น 'สัญญาเช่า' หรือ 'revenue report'...",
)

col1, col2 = st.columns([1, 3])
with col1:
    top_n = st.slider("จำนวนผลลัพธ์", min_value=1, max_value=20, value=5)

if query:
    with st.spinner("🔍 กำลังค้นหา..."):
        try:
            from modules.bm25_search import bm25_search

            results = bm25_search(query, limit=top_n)

            if not results:
                st.info("ℹ️ ไม่พบผลลัพธ์ — ลองเปลี่ยนคำค้นหา")
            else:
                st.markdown(f"### 📋 พบ {len(results)} ผลลัพธ์")
                st.markdown("---")

                for i, result in enumerate(results, 1):
                    source = result.get("source", "unknown")
                    page = result.get("page", "?")
                    text = result.get("text", "")
                    search_source = "BM25"
                    bm25_score = result.get("score", 0)

                    score_display = f"Score: {bm25_score:.3f}"

                    with st.expander(
                        f"**#{i}** 📄 {source} (หน้า {page}) — {score_display}",
                        expanded=(i <= 2),
                    ):
                        # Metadata
                        meta_cols = st.columns(4)
                        meta_cols[0].markdown(f"**ไฟล์:** {source}")
                        meta_cols[1].markdown(f"**หน้า:** {page}")
                        meta_cols[2].markdown(
                            f'<span class="score-badge">{score_display}</span>',
                            unsafe_allow_html=True,
                        )
                        meta_cols[3].markdown(f"**ที่มา:** {search_source}")

                        st.markdown("---")
                        st.markdown(text)

        except Exception as e:
            st.error(f"❌ ค้นหาไม่สำเร็จ: {e}")

# ─── Tips ────────────────────────────────────────────────
with st.expander("💡 เคล็ดลับการค้นหา"):
    st.markdown(
        """
        - ค้นหาเป็นคำ keyword (Keyword Search) จะได้ผลดีที่สุด
        - ใช้ภาษาเดียวกับเอกสาร (ไทย/อังกฤษ) จะได้ผลลัพธ์ดีกว่า
        """
    )
