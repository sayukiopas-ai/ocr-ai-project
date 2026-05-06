"""
Ask Page — RAG Q&A with streaming response.
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

st.markdown("# 💬 ถามคำถามจากเอกสาร")
st.markdown(
    "โหมดเดียว: OCR Document QA — ถามได้ทั้งการเทียบหัวข้อย้อนหลัง "
    "และคำถามอื่นๆ ที่เกี่ยวกับเอกสารในระบบ"
)

st.markdown("---")

# ─── Chat History ────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Welcome message when no chat history
if not st.session_state.chat_history:
    with st.chat_message("assistant"):
        st.markdown(
            """
            <style>
            .typewriter-container {
                position: relative;
                display: inline-block;
            }
            .typewriter-hidden {
                visibility: hidden;
                white-space: nowrap;
                border-right: 0.4em solid transparent; /* match space of cursor */
            }
            .typewriter-text {
                position: absolute;
                top: 0;
                left: 0;
                bottom: 0;
                overflow: hidden;
                white-space: nowrap;
                border-right: 0.4em solid transparent;
                animation: typing 1.5s steps(40, end) forwards, blink-cursor 0.5s step-end 7 forwards;
                width: 0;
            }
            @keyframes typing {
                from { width: 0 }
                to { width: 100% }
            }
            @keyframes blink-cursor {
                from { border-color: rgba(255,255,255,0.7) }
                50%, to { border-color: transparent }
            }
            </style>
            <div class="typewriter-container">
                <div class="typewriter-hidden">สวัสดีครับ วันนี้ให้ช่วยเหลืออะไรดีครับ 😊</div>
                <div class="typewriter-text">สวัสดีครับ วันนี้ให้ช่วยเหลืออะไรดีครับ 😊</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 แหล่งอ้างอิง", expanded=False):
                for j, src in enumerate(msg["sources"]):
                    excerpt = (
                        src.get("text_excerpt")
                        or src.get("text_preview")
                        or ""
                    )
                    st.markdown(
                        f'<div class="source-card">'
                        f'📄 <strong>{src["source"]}</strong> (หน้า {src["page"]}) '
                        f'— <span class="score-badge">Score: {src["score"]:.3f}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    # Download original file
                    import config
                    _fpath = Path(config.UPLOAD_DIR) / src["source"]
                    if _fpath.exists():
                        with open(_fpath, "rb") as _fp:
                            st.download_button(
                                label="📥 เปิดไฟล์ต้นฉบับ",
                                data=_fp,
                                file_name=src["source"],
                                key=f"dl_hist_{id(msg)}_{j}",
                            )
                    if excerpt.strip():
                        st.text(excerpt)
                    else:
                        st.caption("(ไม่มีข้อความใน chunk)")

# ─── Chat Input ──────────────────────────────────────────
question = st.chat_input("ถามคำถามเกี่ยวกับเอกสาร...")

if question:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        try:
            from modules.rag import ask_stream

            answer_parts = []
            sources = []
            placeholder = st.empty()
            status_line = st.empty()

            status_line.markdown("กำลังค้นหาและสรุปคำตอบจากเอกสาร...")
            for event in ask_stream(question, top_n=8):
                if event["type"] == "chunk":
                    answer_parts.append(event["content"])
                    placeholder.markdown("".join(answer_parts) + "▌")
                elif event["type"] == "sources":
                    sources = event["sources"]

            # Final answer without cursor
            full_answer = "".join(answer_parts)
            placeholder.markdown(full_answer)
            status_line.markdown("✅เสร็จสิ้น") 

            # Show sources
            if sources:
                with st.expander("📚 แหล่งอ้างอิง", expanded=False):
                    for j, src in enumerate(sources):
                        excerpt = (
                            src.get("text_excerpt")
                            or src.get("text_preview")
                            or ""
                        )
                        st.markdown(
                            f'<div class="source-card">'
                            f'📄 <strong>{src["source"]}</strong> '
                            f'(หน้า {src["page"]}) '
                            f'— <span class="score-badge">'
                            f'Score: {src["score"]:.3f}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        # Download original file
                        import config
                        _fpath = Path(config.UPLOAD_DIR) / src["source"]
                        if _fpath.exists():
                            with open(_fpath, "rb") as _fp:
                                st.download_button(
                                    label="📥 เปิดไฟล์ต้นฉบับ",
                                    data=_fp,
                                    file_name=src["source"],
                                    key=f"dl_live_{j}",
                                )
                        if excerpt.strip():
                            st.text(excerpt)
                        else:
                            st.caption("(ไม่มีข้อความใน chunk)")

            # Save to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources,
            })

        except Exception as e:
            error_msg = f"❌ ไม่สามารถตอบคำถามได้: {e}"
            st.error(error_msg)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg,
            })

# ─── Clear Chat ──────────────────────────────────────────
if st.session_state.chat_history:
    if st.button("🗑️ ล้างประวัติแชท"):
        st.session_state.chat_history = []
        st.rerun()
