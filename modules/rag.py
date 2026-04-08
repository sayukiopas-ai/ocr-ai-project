"""
RAG pipeline — Retrieve, Augment, Generate using Ollama LLM.
"""

import re
import unicodedata
from typing import Generator

from openai import OpenAI
from rapidfuzz import fuzz

import config
from modules.hybrid_search import hybrid_search

# partial_ratio 0–100 — ค่าหลักสำหรับวลียาว
_THAI_FUZZY_PARTIAL_MIN: int = 86
# วลีสั้น (หัวข้อ รายงาน / ชื่อย่อ) ให้ยืดหยุ่นกว่าเล็กน้อย
_THAI_FUZZY_PARTIAL_SHORT: int = 72  # ความยาวชิ้นหลัง normalize ไม่เกิน 7 ตัว
_THAI_FUZZY_PARTIAL_MED: int = 80  # ความยาว 8–12

# Minimum rerank score to keep a chunk — filters out clearly irrelevant docs.
_MIN_RERANK_SCORE: float = 0.15

SYSTEM_PROMPT = """คุณเป็นผู้ช่วย OCR Document QA สำหรับเอกสารองค์กร
กฎสำคัญ:
1. ใช้เฉพาะ context เอกสารที่ให้มาเท่านั้น ห้ามแต่งข้อมูลและห้ามใช้ความรู้ภายนอก
2. ถ้าข้อมูลไม่พอ ให้ตอบชัดเจนว่า "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร"
3. ตอบเป็นภาษาเดียวกับคำถาม (ไทย/อังกฤษ)
4. คำตอบต้องกระชับ ตรงประเด็น และอ้างอิงชื่อไฟล์/หน้าที่เกี่ยวข้องเมื่อทำได้

แนวทางการตอบ:
5. ถ้าคำถามต้องการ "เทียบหัวข้อกับอดีต" หรือ "มีหัวข้อไหนคล้ายกัน":
   - สรุปหัวข้อที่เกี่ยวข้องเป็นรายการ
   - ในแต่ละรายการให้มี: หัวข้อ/ชื่อเรื่อง, เนื้อหาโดยย่อ, เหตุผลที่เกี่ยวข้อง
   - ถ้ามีวันที่/หมวด ให้ใส่เพิ่ม
6. ถ้าคำถามเป็นการดึงข้อมูลเฉพาะ (วันที่, หมวด, ชื่อเรื่อง, เลขที่เอกสาร ฯลฯ):
   - ให้ตอบค่าที่ถามโดยตรงก่อน
   - ถ้าพบหลายค่า ให้สรุปเป็น bullet สั้นๆ
   - ถ้าไม่พบ field ที่ถาม ให้บอกตรงๆ ว่าไม่พบ field นั้น
7. ถ้า context เจอหลายเอกสารที่ใกล้เคียงกัน ให้เรียงตามความเกี่ยวข้องจากมากไปน้อย
8. ถ้า context มีข้อความที่ตอบคำถามได้แม้สั้นมาก ให้สรุป/อ้างอิงข้อความนั้น ห้ามตอบว่าไม่พบ"""

GENERAL_SYSTEM_PROMPT = """คุณเป็นผู้ช่วย AI ทั่วไปที่ตอบคำถามได้หลากหลายหัวข้อ
กฎ:
1. ตอบให้ถูกต้อง กระชับ และเข้าใจง่าย
2. ถ้าไม่แน่ใจ ให้บอกอย่างตรงไปตรงมา
3. ตอบเป็นภาษาเดียวกับคำถาม (ไทย/อังกฤษ)"""


def _get_llm_client() -> OpenAI:
    """Create OpenAI-compatible client pointing at Ollama."""
    return OpenAI(
        api_key="ollama",  # Ollama ignores the key but the SDK requires one
        base_url=f"{config.OLLAMA_BASE_URL}/v1",
    )


def _extract_keyword_signals(question: str) -> tuple[str, list[str]]:
    """
    Pull English/number tokens from the question for strict evidence matching.
    Thai-only questions yield empty tokens → caller should fall back to all chunks.
    """
    q = question.lower().strip()
    tokens = re.findall(r"[a-z0-9]+", q)
    tokens = [t for t in tokens if len(t) >= 2]
    phrase = " ".join(tokens)
    return phrase, tokens


_THAI_FILLERS = frozenset(
    {
        "ค้นหา",
        "ค้น",
        "หา",
        "ข้อมูล",
        "ของ",
        "เกี่ยวกับ",
        "เรื่อง",
        "ใน",
        "ที่",
        "และ",
        "หรือ",
        "กับ",
        "จาก",
        "กรุณา",
        "บอก",
        "ให้",
        # มักเกิดเมื่อพิมพ์/ตัดคำว่า "ค้นหา" ผิด
        "นหา",
    }
)


def _extract_thai_fragments(question: str) -> list[str]:
    """
    Pull Thai keyword fragments so evidence filtering works for Thai queries.
    """
    q = question.strip()
    if not q:
        return []

    frags: list[str] = []
    for part in re.split(r"\s+", q):
        part = part.strip()
        if len(part) < 2 or part in _THAI_FILLERS:
            continue
        if re.search(r"[\u0e00-\u0e7f]", part):
            frags.append(part)

    for m in re.finditer(r"[\u0e00-\u0e7f]{2,}", q):
        s = m.group(0)
        if s not in _THAI_FILLERS and s not in frags:
            frags.append(s)

    seen: set[str] = set()
    out: list[str] = []
    for f in frags:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _normalize_for_fuzzy_match(s: str) -> str:
    """Unicode + whitespace normalization for Thai / mixed-string comparison."""
    t = unicodedata.normalize("NFC", s)
    t = re.sub(r"[\u200b-\u200d\ufeff]", "", t)
    # Strip separators between Thai chars for abbreviation matching
    # e.g. พ.ร.บ. → พรบ, พ-ร-บ → พรบ
    t = re.sub(r'(?<=[\u0e00-\u0e7f])[.\-/](?=[\u0e00-\u0e7f])', '', t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def _split_thai_query_into_pieces(frag: str) -> list[str]:
    """
    แตกวลีค้นหาที่ยาวเป็นหัว+หาง เช่น "รายงานปฏิบัติงาน" ที่ในเอกสารเป็น
    "รายงานผลการปฏิบัติงาน..." (มีคำคั่น) — ให้แมตช์แบบ AND รายงาน+ปฏิบัติงาน
    """
    f = frag.strip()
    if len(f) < 10:
        return [f]
    for head in ("รายงาน", "รายการ", "ผลการ"):
        if f.startswith(head) and len(f) > len(head) + 2:
            tail = f[len(head) :].strip()
            if len(tail) >= 3:
                return [head, tail]
    return [f]


def _min_partial_threshold(piece: str) -> int:
    n = len(_normalize_for_fuzzy_match(piece))
    if n <= 7:
        return _THAI_FUZZY_PARTIAL_SHORT
    if n <= 12:
        return _THAI_FUZZY_PARTIAL_MED
    return _THAI_FUZZY_PARTIAL_MIN


def _thai_single_piece_matches(piece: str, text_raw: str) -> bool:
    """แมตช์ชิ้นเดียว: substring ก่อน แล้วจึง fuzzy partial ตามความยาว"""
    f = _normalize_for_fuzzy_match(piece)
    t = _normalize_for_fuzzy_match(text_raw)
    if not f:
        return True
    if not t:
        return False
    if f in t:
        return True
    if len(f) < 4:
        return False
    need = _min_partial_threshold(piece)
    return int(fuzz.partial_ratio(f, t)) >= need


def _thai_frag_matches_text(frag: str, text_raw: str) -> bool:
    """
    หลายชิ้นต้องผ่านทุกชิ้น (AND) — ลดกรณีวลีติดกันใน query แต่ในเอกสารมีคำอื่นคั่น
    """
    pieces = _split_thai_query_into_pieces(frag)
    return all(_thai_single_piece_matches(p, text_raw) for p in pieces)


def _chunk_has_query_evidence(chunk: dict, question: str) -> bool:
    """True if chunk text or filename plausibly matches keyword intent."""
    source_name = str(chunk.get("source", "")).lower()
    text_raw = str(chunk.get("text", ""))
    text = text_raw.lower()
    search_source = str(chunk.get("search_source", ""))

    # Explicit filename in the question → match payload source (OCR text often omits it).
    for m in re.finditer(
        r"[\w./-]+\.(?:pdf|png|jpe?g|gif|webp|bmp|tiff?|docx|xlsx)\b",
        question,
        re.IGNORECASE,
    ):
        fname = m.group(0).lower().split("/")[-1]
        if fname in source_name or source_name.endswith(fname) or fname in source_name.replace(" ", ""):
            return True

    thai_frags = _extract_thai_fragments(question)
    phrase, tokens = _extract_keyword_signals(question)
    has_thai = bool(thai_frags)
    has_eng = bool(phrase or tokens)
    if not has_thai and not has_eng:
        # Query is all filler words (e.g. "มีอะไรบ้าง") — can't filter,
        # accept every chunk and trust search + reranker.
        return True

    thai_ok = (
        all(
            _thai_frag_matches_text(frag, text_raw)
            or _thai_frag_matches_text(frag, source_name)
            for frag in thai_frags
        )
        if has_thai
        else True
    )

    eng_ok = True
    if has_eng:
        eng_ok = False
        # Filename-targeted retrieval from hybrid_search
        if search_source == "source":
            compact = source_name.replace("_", " ")
            if phrase and (phrase in compact or phrase in source_name):
                eng_ok = True
            if not eng_ok and tokens and all(
                t in compact or t in source_name for t in tokens
            ):
                eng_ok = True
            if not eng_ok and any(
                len(t) >= 3 and (t in compact or t in source_name) for t in tokens
            ):
                eng_ok = True

        # Exact phrase in OCR/body (e.g. "scan me")
        if not eng_ok and phrase and len(phrase) >= 3:
            if phrase in text:
                eng_ok = True
            elif phrase.replace(" ", "") in text.replace(" ", ""):
                eng_ok = True

        # Multi-token: require every token to appear (reduces false positives)
        if not eng_ok and len(tokens) >= 2:
            eng_ok = all(t in text for t in tokens)
        if not eng_ok and len(tokens) == 1:
            eng_ok = tokens[0] in text

    if has_thai and has_eng:
        return thai_ok and eng_ok
    if has_thai:
        return thai_ok
    return eng_ok


def _filter_chunks_by_evidence(chunks: list[dict], question: str) -> list[dict]:
    """
    Filter chunks that lack keyword evidence from the query.

    - If query has extractable keywords but no chunk matches → return []
      so the LLM correctly says "ไม่พบข้อมูลที่เกี่ยวข้อง".
    - If query has NO extractable keywords (all fillers) → skip filtering
      and trust hybrid-search + reranker.
    """
    if not chunks:
        return chunks

    filtered = [c for c in chunks if _chunk_has_query_evidence(c, question)]
    return filtered


def _filter_low_score_chunks(chunks: list[dict]) -> list[dict]:
    """
    Remove chunks whose rerank_score falls below _MIN_RERANK_SCORE.
    Prevents clearly irrelevant documents from polluting context & intro.
    """
    if not chunks:
        return chunks
    filtered = [
        c for c in chunks
        if c.get("rerank_score", c.get("score", 0)) >= _MIN_RERANK_SCORE
    ]
    return filtered


def _build_context(chunks: list[dict]) -> str:
    """Build context string from retrieved chunks."""
    if not chunks:
        return "ไม่พบเอกสารที่เกี่ยวข้อง"

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        title = chunk.get("title")
        category = chunk.get("category")
        date = chunk.get("date")

        meta_parts = []
        if title:
            meta_parts.append(f"title={title}")
        if category:
            meta_parts.append(f"category={category}")
        if date:
            meta_parts.append(f"date={date}")
        meta_line = f" | metadata: {', '.join(meta_parts)}" if meta_parts else ""

        context_parts.append(
            f"[เอกสาร {i}] ไฟล์: {source} | หน้า: {page}{meta_line}\n{text}"
        )

    return "\n\n---\n\n".join(context_parts)


def _build_sources(chunks: list[dict]) -> list[dict]:
    """Build normalized source references for UI and answer preface."""
    return [
        {
            "source": c.get("source", "unknown"),
            "page": c.get("page", "?"),
            "score": c.get("rerank_score", c.get("score", 0)),
            "search_source": c.get("search_source", ""),
            "text_preview": c.get("text", "")[:200],
            "text_excerpt": (c.get("text") or "")[:4000],
        }
        for c in chunks
    ]


def _build_found_docs_intro(
    chunks: list[dict],
    max_items: int = 3,
) -> str:
    """
    Build a deterministic intro listing found files/pages before LLM summary.
    Expects `chunks` to already be evidence-filtered when keywords exist.
    """
    if not chunks:
        return ""

    seen: set[tuple[str, str]] = set()
    items: list[str] = []
    for c in chunks:
        source = str(c.get("source", "unknown"))
        page = str(c.get("page", "?"))
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        items.append(f"- {source} (หน้า {page})")
        if len(items) >= max_items:
            break

    if not items:
        return ""
    return "พบเอกสารที่เกี่ยวข้อง:\n" + "\n".join(items) + "\n\n"


def _build_retrieved_excerpts_markdown(
    chunks: list[dict],
    *,
    max_chunks: int = 8,
    max_chars_per_chunk: int = 4000,
) -> str:
    """
    Show raw OCR / chunk text so users see file contents even if the LLM is overly cautious.
    """
    if not chunks:
        return ""

    parts: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    for c in chunks[:max_chunks]:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        source = str(c.get("source", "unknown"))
        page = str(c.get("page", "?"))
        dedupe_key = (source, page, text[:120])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        snippet = text
        if len(snippet) > max_chars_per_chunk:
            snippet = snippet[:max_chars_per_chunk] + "…"
        # Avoid breaking fenced code blocks in Markdown renderers
        snippet = snippet.replace("```", "'''")

        parts.append(f"**📄 {source}** · หน้า {page}\n\n```\n{snippet}\n```")

    if not parts:
        return ""

    return (
        "\n\n---\n\n"
        "### 📑 เนื้อหาที่ดึงจากเอกสาร (ข้อความจาก OCR / chunk)\n\n"
        + "\n\n".join(parts)
    )


def ask(
    question: str,
    top_n: int = 5,
) -> dict:
    """
    Full RAG pipeline: retrieve → augment → generate.

    Args:
        question: User's question.
        top_n: Number of chunks to retrieve.

    Returns:
        {"answer": str, "sources": list[dict]}
    """
    # 1. Retrieve
    chunks = hybrid_search(question, final_top_n=top_n)
    chunks = _filter_chunks_by_evidence(chunks, question)
    chunks = _filter_low_score_chunks(chunks)

    # 2. Augment
    context = _build_context(chunks)

    # 3. Generate
    client = _get_llm_client()
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context จากเอกสาร:\n\n{context}\n\n"
                    f"คำถาม: {question}"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    answer = response.choices[0].message.content.strip()
    sources = _build_sources(chunks)
    intro = _build_found_docs_intro(chunks)
    excerpts = _build_retrieved_excerpts_markdown(chunks)
    if intro:
        answer = intro + answer
    if excerpts:
        answer = answer + excerpts

    return {"answer": answer, "sources": sources}


def ask_stream(
    question: str,
    top_n: int = 5,
) -> Generator[dict, None, None]:
    """
    Streaming RAG — yields partial answers as they arrive.

    Yields:
        {"type": "chunk", "content": str}  — partial answer text
        {"type": "sources", "sources": list}  — source references (final)
    """
    # 1. Retrieve
    chunks = hybrid_search(question, final_top_n=top_n)
    chunks = _filter_chunks_by_evidence(chunks, question)
    chunks = _filter_low_score_chunks(chunks)

    # 2. Augment
    context = _build_context(chunks)

    sources = _build_sources(chunks)
    intro = _build_found_docs_intro(chunks)
    if intro:
        yield {"type": "chunk", "content": intro}

    # 3. Stream generate
    client = _get_llm_client()
    stream = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context จากเอกสาร:\n\n{context}\n\n"
                    f"คำถาม: {question}"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=2048,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield {"type": "chunk", "content": chunk.choices[0].delta.content}

    excerpts = _build_retrieved_excerpts_markdown(chunks)
    if excerpts:
        yield {"type": "chunk", "content": excerpts}

    # Yield sources at the end
    yield {"type": "sources", "sources": sources}


def ask_general_stream(question: str) -> Generator[dict, None, None]:
    """
    General chat mode — no document retrieval, LLM only.

    Yields:
        {"type": "chunk", "content": str}
    """
    client = _get_llm_client()
    stream = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.4,
        max_tokens=2048,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield {"type": "chunk", "content": chunk.choices[0].delta.content}
