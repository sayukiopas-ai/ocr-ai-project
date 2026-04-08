import streamlit as st

def show_sidebar():
    with st.sidebar:
        st.markdown("## 📚 OCR Knowledge Base")
        st.markdown("---")
        st.markdown("**เมนู**")
        st.page_link("app.py", label="Home (Upload)", icon="🏠")
        st.page_link("pages/2_🔍_Search.py", label="Search (ค้นหา)", icon="🔍")
        st.page_link("pages/3_💬_Ask.py", label="Ask (ถาม-ตอบ)", icon="💬")
        st.page_link("pages/4_📚_Library.py", label="Library (คลังเอกสาร)", icon="📚")
        st.markdown("---")
        
        # Show collection stats
        try:
            from modules.vector_store import get_collection_info
            info = get_collection_info()
            st.metric("📊 Documents Indexed", info.get("points_count", 0))
            st.caption(f"Status: {'✅' if info.get('status') == 'green' else '❌'}")
        except Exception:
            st.caption("⚠️ Qdrant not connected")
