"""
Library Page — view and manage uploaded documents with Qdrant sync status.
"""

import os
import streamlit as st
from pathlib import Path

import config
from modules.vector_store import (
    delete_by_source,
    get_sources_stats,
    count_by_source,
    invalidate_cache,
)
from modules.sidebar import show_sidebar

# Load CSS
css_path = Path(__file__).parent.parent / "styles" / "custom.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Show custom sidebar
show_sidebar()

st.markdown("# 📚 Library (คลังเอกสาร)")
st.markdown("ดูและจัดการเอกสารทั้งหมดที่เคยอัปโหลดไว้ในระบบ")
st.markdown("---")

# Session state to track which file is pending deletion confirmation
if "delete_confirm" not in st.session_state:
    st.session_state.delete_confirm = None

upload_dir = Path(config.UPLOAD_DIR)

# Ensure directory exists
if not upload_dir.exists():
    upload_dir.mkdir(parents=True, exist_ok=True)

# ─── Gather data from both sources ────────────────────────
# 1. Files on disk
disk_files = list(upload_dir.glob("*"))
disk_files = [f for f in disk_files if f.is_file() and f.name != ".gitkeep"]
disk_filenames = {f.name for f in disk_files}

# 2. All sources indexed in Qdrant (force fresh data)
invalidate_cache()
try:
    qdrant_sources = get_sources_stats()  # {source_name: chunk_count}
except Exception:
    qdrant_sources = {}

qdrant_filenames = set(qdrant_sources.keys())

# Combine: union of both sets
all_filenames = sorted(disk_filenames | qdrant_filenames)

if not all_filenames:
    st.info("📁 ขณะนี้ยังไม่มีเอกสารในคลัง")
    st.stop()

# ─── Summary ──────────────────────────────────────────────
col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("📄 ไฟล์ทั้งหมด", len(all_filenames))
col_s2.metric("💾 ไฟล์บน Disk", len(disk_filenames))
col_s3.metric("🧠 แหล่งใน Qdrant", len(qdrant_filenames))

# Detect mismatches
only_on_disk = disk_filenames - qdrant_filenames
only_in_qdrant = qdrant_filenames - disk_filenames

if only_on_disk:
    with st.expander(f"⚠️ ไฟล์ที่ยังไม่ได้ Index ({len(only_on_disk)} ไฟล์)", expanded=False):
        st.caption("ไฟล์เหล่านี้อยู่บน disk แต่ยังไม่มี chunk ใน Qdrant — กรุณา Upload & Process ใหม่")
        for name in sorted(only_on_disk):
            st.markdown(f"- `{name}`")

if only_in_qdrant:
    with st.expander(f"⚠️ ข้อมูลใน Qdrant ที่ไม่มีไฟล์ต้นฉบับ ({len(only_in_qdrant)} แหล่ง)", expanded=False):
        st.caption("ข้อมูลเหล่านี้อยู่ใน Qdrant แต่ไฟล์ต้นฉบับถูกลบหรือเปลี่ยนชื่อแล้ว")
        for name in sorted(only_in_qdrant):
            st.markdown(f"- `{name}` ({qdrant_sources.get(name, 0)} chunks)")

st.markdown("---")
st.markdown(f"**📋 รายการเอกสารทั้งหมด ({len(all_filenames)} ไฟล์)**")
st.markdown("<br>", unsafe_allow_html=True)

# ─── Document List ────────────────────────────────────────
for fname in all_filenames:
    on_disk = fname in disk_filenames
    in_qdrant = fname in qdrant_filenames
    chunk_count = qdrant_sources.get(fname, 0)

    col1, col2, col3, col4 = st.columns([4, 2, 2, 2])

    # File info
    col1.markdown(f"**📄 {fname}**")

    # File size (if on disk)
    if on_disk:
        disk_file = upload_dir / fname
        size_kb = disk_file.stat().st_size / 1024
        col2.caption(f"📂 {size_kb:.1f} KB")
    else:
        col2.caption("📂 ❌ ไม่พบไฟล์")

    # Qdrant status
    if in_qdrant:
        col3.caption(f"🧠 {chunk_count} chunks")
    else:
        col3.caption("🧠 ❌ ไม่มีใน Qdrant")

    # Delete action logic
    if st.session_state.delete_confirm == fname:
        with col4:
            pass  # Button is placed below

        # Show confirmation below the row
        st.warning(f"⚠️ ยืนยันการลบ **{fname}**? (จะลบเฉพาะเอกสารนี้เท่านั้น ไม่กระทบเอกสารอื่น)")

        cc1, cc2 = st.columns(2)

        # Show what will be deleted
        delete_details = []
        if on_disk:
            delete_details.append("✅ ลบไฟล์จาก disk")
        if in_qdrant:
            delete_details.append(f"✅ ลบ {chunk_count} chunks จาก Qdrant")
        if delete_details:
            st.caption(" | ".join(delete_details))

        if cc1.button("🗑️ ยืนยันลบ", key=f"confirm_{fname}", type="primary"):
            try:
                deleted_chunks = 0

                # 1. Delete from Qdrant vector store (only this source)
                if in_qdrant:
                    deleted_chunks = delete_by_source(fname)

                # 2. Delete physical file (only this file)
                if on_disk:
                    (upload_dir / fname).unlink()

                # 3. Invalidate cache so next load is fresh
                invalidate_cache()

                # Reset confirm state
                st.session_state.delete_confirm = None
                st.toast(
                    f"✅ ลบเอกสาร {fname} เรียบร้อยแล้ว "
                    f"(ลบ {deleted_chunks} chunks จาก Qdrant)"
                )
                st.rerun()
            except Exception as e:
                st.error(f"❌ ลบไม่สำเร็จ: {e}")

        if cc2.button("ยกเลิก", key=f"cancel_{fname}"):
            st.session_state.delete_confirm = None
            st.rerun()
    else:
        if col4.button("🗑️ ลบ", key=f"btn_{fname}"):
            st.session_state.delete_confirm = fname
            st.rerun()

    st.markdown("---")
