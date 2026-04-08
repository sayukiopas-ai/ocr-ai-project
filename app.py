"""
Upload & OCR Page — handles file upload, text extraction, and indexing.
"""

import streamlit as st
from pathlib import Path
import shutil

# Load CSS
css_path = Path(__file__).parent / "styles" / "custom.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Show custom sidebar
from modules.sidebar import show_sidebar
show_sidebar()

st.markdown("# 🏠 Home / 📄 Upload & OCR")
st.markdown("อัปโหลดเอกสาร แล้วระบบจะดึงข้อความ, ตัดเป็น chunks, และจัดเก็บเข้า Knowledge Base")

st.markdown("---")

# ─── File Uploader ───────────────────────────────────────
uploaded_files = st.file_uploader(
    "เลือกไฟล์ (รองรับ PDF, DOCX, XLSX, PNG, JPG)",
    type=["pdf", "docx", "xlsx", "png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "tif"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.markdown(f"**📁 เลือกแล้ว {len(uploaded_files)} ไฟล์**")

    if st.button("🚀 เริ่ม Process", type="primary"):
        import config
        from modules import file_parser, ocr, chunker, embedder, vector_store

        for uploaded_file in uploaded_files:
            st.markdown(f"---\n### 📎 {uploaded_file.name}")
            progress = st.progress(0, text="กำลังบันทึกไฟล์...")

            # 1. Save file
            save_path = Path(config.UPLOAD_DIR) / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            progress.progress(10, text="บันทึกไฟล์แล้ว ✓")

            # 2. Extract text
            progress.progress(15, text="กำลังดึงข้อความ...")
            try:
                parsed = file_parser.extract_text(str(save_path))
            except Exception as e:
                st.error(f"❌ ดึงข้อความไม่สำเร็จ: {e}")
                continue

            progress.progress(30, text="ดึงข้อความแล้ว ✓")

            # 3. OCR if needed
            pages: list[dict] = []

            if parsed.get("needs_ocr"):
                progress.progress(35, text="🔍 กำลัง OCR ด้วย Typhoon Vision...")
                try:
                    if parsed["type"] == "image":
                        ocr_text = ocr.ocr_image(str(save_path))
                        pages = [{"page": 1, "text": ocr_text}]
                    elif parsed["type"] == "pdf":
                        def _ocr_progress(cur, total):
                            pct = 35 + int((cur / max(total, 1)) * 25)
                            progress.progress(
                                min(pct, 60),
                                text=f"🔍 OCR หน้า {cur}/{total}...",
                            )
                        pages = ocr.ocr_pdf(str(save_path), progress_callback=_ocr_progress)
                    progress.progress(60, text="OCR เสร็จแล้ว ✓")
                except Exception as e:
                    st.error(f"❌ OCR ไม่สำเร็จ: {e}")
                    continue
            else:
                # Use extracted text
                if "pages" in parsed:
                    pages = parsed["pages"]
                elif "text" in parsed:
                    pages = [{"page": 1, "text": parsed["text"]}]
                progress.progress(60, text="ใช้ข้อความที่ดึงได้ ✓")

            # Clean Thai spacing artifacts (e.g. spaces before upper/lower vowels out of PDF/OCR)
            import re
            for p in pages:
                p["text"] = re.sub(r'\s+([ัิีึืุู็์ํ่้๊๋])', r'\1', p["text"])

            # Show extracted text preview
            total_chars = sum(len(p["text"]) for p in pages)
            with st.expander(f"👁️ ดูข้อความที่ดึงได้ ({total_chars:,} ตัวอักษร)"):
                for p in pages[:5]:  # Show max 5 pages
                    st.markdown(f"**หน้า {p['page']}:**")
                    st.text(p["text"][:500] + ("..." if len(p["text"]) > 500 else ""))

            # 4. Chunk
            progress.progress(65, text="กำลังตัด chunks...")
            chunks = chunker.chunk_pages(pages)
            progress.progress(70, text=f"ตัดได้ {len(chunks)} chunks ✓")

            if not chunks:
                st.warning("⚠️ ไม่มีข้อความให้จัดเก็บ")
                continue

            # 5. Embed
            progress.progress(75, text="กำลังสร้าง embeddings (Ollama)...")
            try:
                texts = [c["text"] for c in chunks]

                def _embed_progress(done, total):
                    pct = 75 + int((done / max(total, 1)) * 10)
                    progress.progress(
                        min(pct, 85),
                        text=f"Embedding {done}/{total} chunks...",
                    )

                embeddings = embedder.embed_texts(texts, progress_callback=_embed_progress)
                progress.progress(85, text=f"Embedding เสร็จ ({len(embeddings)} vectors) ✓")
            except Exception as e:
                st.error(f"❌ Embedding ไม่สำเร็จ: {e}")
                continue

            # 6. Store in Qdrant
            progress.progress(90, text="กำลังจัดเก็บใน Qdrant...")
            try:
                # Delete old version if exists
                vector_store.delete_by_source(uploaded_file.name)

                point_ids = vector_store.upsert_chunks(
                    chunks=chunks,
                    embeddings=embeddings,
                    metadata={"source": uploaded_file.name},
                )
                progress.progress(100, text="✅ เสร็จสิ้น!")
            except Exception as e:
                st.error(f"❌ จัดเก็บไม่สำเร็จ: {e}")
                continue

            st.success(
                f"✅ **{uploaded_file.name}** — "
                f"{len(chunks)} chunks, {len(embeddings)} vectors, "
                f"{total_chars:,} ตัวอักษร"
            )

        st.balloons()

# ─── Currently Indexed ───────────────────────────────────
st.markdown("---")
st.markdown("### 📊 สถานะ Knowledge Base")

try:
    from modules.vector_store import get_collection_info
    info = get_collection_info()
    col1, col2, col3 = st.columns(3)
    col1.metric("Chunks ทั้งหมด", info.get("points_count", 0))
    col2.metric("Vectors", info.get("vectors_count", 0))
    col3.metric("Status", "✅" if info.get("status") == "green" else "❌")
except Exception as e:
    st.info(f"ℹ️ ยังไม่ได้เชื่อมต่อ Qdrant — {e}")
