"""Topic discovery and evidence evaluation for topic videos."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import re
import sqlite3
from typing import Any

from streamclipper.config import Config
from streamclipper.library.index import LibraryIndex
from streamclipper.topics.ledger import TopicLedger
from streamclipper.topics.models import EvidenceRef, TopicCandidate

logger = logging.getLogger(__name__)

# Stopwords and generic filler to remove from salient term extraction
EN_STOPWORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing",
    "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "let", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
    "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
}

HINGLISH_STOPWORDS: set[str] = {
    "bhai", "bro", "yaar", "kya", "hai", "yeh", "woh", "bhi", "toh", "ka", "ki", "ke", "ko", "se",
    "me", "mein", "par", "pe", "nahi", "tha", "thi", "the", "kar", "karo", "raha", "rahi", "rahe",
    "ab", "kab", "jab", "aur", "lekin", "magar", "kyun", "kaise", "hoga", "hogi", "accha", "theek",
    "arre", "aree", "are", "abe", "haan", "nahin", "apna", "apne", "hum", "tum", "mera", "meri",
    "tere", "teri", "tera", "unka", "inka", "wala", "wali", "wale", "kuch", "sab", "bohot", "bahut",
    "karna", "karte", "karta", "karti", "gaya", "gayi", "gaye", "aaya", "aayi", "aaye", "dekh",
    "dekho", "dekha", "bol", "bolo", "bola", "sun", "suno", "suna",
}

GENERIC_GAMING_TALK: set[str] = {
    "chat", "guys", "lol", "lmao", "gg", "game", "gameplay", "stream", "streamer", "live", "play",
    "playing", "player", "look", "see", "oh", "okay", "yeah", "yes", "no", "wait", "go", "going",
    "gone", "come", "coming", "just", "like", "get", "got", "getting", "one", "two", "three",
    "take", "taking", "make", "making", "think", "know", "watch", "right", "now", "well", "gonna",
    "wanna", "gotta", "dude", "man", "video", "clip", "subscribe", "comment", "share", "channel",
    "actually", "literally", "basically", "really", "something", "someone", "anyone", "anything",
    "nothing", "everything", "always", "never", "maybe", "probably", "definitely", "omg", "bruh",
}

ALL_STOPWORDS: set[str] = EN_STOPWORDS | HINGLISH_STOPWORDS | GENERIC_GAMING_TALK


def _slugify(text: str) -> str:
    """Convert a title to a clean URL/file-safe slug."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s or "topic"


def _format_time(seconds: float) -> str:
    """Format seconds into MM:SS format."""
    sec = max(0.0, float(seconds))
    mm = int(sec // 60)
    ss = int(sec % 60)
    return f"{mm:02d}:{ss:02d}"


def _extract_snippet_window(text: str, term: str, max_chars: int = 140) -> str:
    """Extract a window centered on term so the key phrase is not truncated in snippets."""
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    term_lower = term.lower()
    clean_lower = clean.lower()
    idx = clean_lower.find(term_lower)
    if idx == -1:
        first_word = term_lower.split()[0]
        idx = clean_lower.find(first_word)
    if idx == -1:
        return clean[: max_chars - 3] + "..."
    term_len = len(term)
    half = (max_chars - term_len) // 2
    start = max(0, idx - half)
    end = min(len(clean), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)
    snippet = clean[start:end].strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(clean) else ""
    return f"{prefix}{snippet}{suffix}"


def _extract_json(raw_text: str) -> Any:
    """Robustly parse JSON data from LLM responses, handling fences, thinking tags, and conversational prefixes."""
    cleaned = re.sub(r"<think>.*?</think>", "", str(raw_text or ""), flags=re.DOTALL | re.IGNORECASE).strip()
    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except Exception:
            pass

    if cleaned.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
        stripped = re.sub(r"\n?```\s*$", "", stripped).strip()
        try:
            return json.loads(stripped)
        except Exception:
            pass

    array_match = re.search(r"(\[.*\])", cleaned, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(1).strip())
        except Exception:
            pass

    obj_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(1).strip())
        except Exception:
            pass

    return None


def topic_material(
    config: Config,
    game: str,
    *,
    max_terms: int = 40,
) -> dict[str, Any]:
    """Extract top salient terms/bigrams and snippets across a game's library segments and frame notes.

    Uses TF-IDF style scoring: terms frequent across multiple windows and sources, but rare in
    generic gaming talk, with stopwords and gaming filler removed.
    """
    index = LibraryIndex(config)
    ledger = TopicLedger(config)
    existing_entries = ledger.entries(game)
    existing_titles = [str(e.get("title", "")) for e in existing_entries if e.get("title")]

    sources = index.list_sources(game=game)
    source_map = {s.source_id: s for s in sources}

    stats = {
        "sources": len(sources),
        "channels": len({s.channel for s in sources if s.channel}),
        "kinds": sorted(list({s.kind for s in sources if s.kind})),
        "total_hours": round(sum(s.duration_seconds for s in sources) / 3600.0, 2),
    }

    if not source_map:
        return {
            "game": game,
            "stats": stats,
            "existing_titles": existing_titles,
            "terms": [],
        }

    # Gather documents (segments + frame notes)
    documents: list[dict[str, Any]] = []

    with index._get_connection() as conn:
        if game and str(game).strip():
            seg_rows = conn.execute(
                """
                SELECT s.id, s.source_id, s.start, s.end, s.text, s.chat_text, src.channel
                FROM segments s
                JOIN sources src ON s.source_id = src.source_id
                WHERE LOWER(src.game) = LOWER(?);
                """,
                [str(game).strip()],
            ).fetchall()
        else:
            seg_rows = conn.execute(
                """
                SELECT s.id, s.source_id, s.start, s.end, s.text, s.chat_text, src.channel
                FROM segments s
                JOIN sources src ON s.source_id = src.source_id;
                """
            ).fetchall()

        for r in seg_rows:
            spoken = str(r["text"] or "").strip()
            chat = str(r["chat_text"] or "").strip()
            combined_text = f"{spoken} {chat}".strip() if spoken else chat
            if combined_text:
                ch = str(r["channel"] or "")
                documents.append({
                    "doc_id": f"seg_{r['id']}",
                    "source_id": str(r["source_id"]),
                    "channel": ch,
                    "time": float(r["start"]),
                    "text": combined_text,
                })

        try:
            if game and str(game).strip():
                fn_rows = conn.execute(
                    """
                    SELECT fn.id, fn.source_id, fn.t, fn.note, src.channel
                    FROM frame_notes fn
                    JOIN sources src ON fn.source_id = src.source_id
                    WHERE LOWER(src.game) = LOWER(?);
                    """,
                    [str(game).strip()],
                ).fetchall()
            else:
                fn_rows = conn.execute(
                    """
                    SELECT fn.id, fn.source_id, fn.t, fn.note, src.channel
                    FROM frame_notes fn
                    JOIN sources src ON fn.source_id = src.source_id;
                    """
                ).fetchall()

            for r in fn_rows:
                note_text = str(r["note"] or "").strip()
                if note_text:
                    ch = str(r["channel"] or "")
                    documents.append({
                        "doc_id": f"fn_{r['id']}",
                        "source_id": str(r["source_id"]),
                        "channel": ch,
                        "time": float(r["t"]),
                        "text": note_text,
                    })
        except sqlite3.OperationalError:
            pass

    if not documents:
        return {
            "game": game,
            "stats": stats,
            "existing_titles": existing_titles,
            "terms": [],
        }

    # Extract terms, bigrams, and pairwise co-occurrences from each document
    term_doc_count: dict[str, int] = {}
    term_source_count: dict[str, set[str]] = {}
    term_total_count: dict[str, int] = {}
    term_snippets: dict[str, list[dict[str, Any]]] = {}
    cooccurrence: dict[str, dict[str, int]] = {}

    word_pattern = re.compile(r"[a-zA-Z0-9\u0900-\u097F]+(?:'[a-zA-Z0-9]+)?")

    for doc in documents:
        raw_words = [w.lower() for w in word_pattern.findall(doc["text"])]
        valid_words: list[str] = []
        for w in raw_words:
            if len(w) >= 3 and w not in ALL_STOPWORDS and not w.isdigit():
                valid_words.append(w)

        # Build bigrams from adjacent valid words
        doc_terms: set[str] = set()
        for w in valid_words:
            doc_terms.add(w)
            term_total_count[w] = term_total_count.get(w, 0) + 1

        for i in range(len(valid_words) - 1):
            bg = f"{valid_words[i]} {valid_words[i+1]}"
            doc_terms.add(bg)
            term_total_count[bg] = term_total_count.get(bg, 0) + 1

        doc_terms_list = list(doc_terms)
        for term in doc_terms:
            term_doc_count[term] = term_doc_count.get(term, 0) + 1
            term_source_count.setdefault(term, set()).add(doc["source_id"])

            # Collect diverse candidate snippets across channels / sources (up to 3 per channel, max 15)
            snippets_list = term_snippets.setdefault(term, [])
            ch = doc.get("channel") or doc.get("source_id") or "Unknown"
            ch_count = sum(1 for d in snippets_list if (d.get("channel") or d.get("source_id") or "Unknown") == ch)
            if ch_count < 3 and len(snippets_list) < 15:
                snippets_list.append(doc)

        # Track pairwise co-occurrences in this document
        for i in range(len(doc_terms_list)):
            t1 = doc_terms_list[i]
            co_map = cooccurrence.setdefault(t1, {})
            for j in range(i + 1, len(doc_terms_list)):
                t2 = doc_terms_list[j]
                co_map[t2] = co_map.get(t2, 0) + 1
                cooccurrence.setdefault(t2, {})[t1] = cooccurrence.setdefault(t2, {}).get(t1, 0) + 1

    n_docs = len(documents)
    scored_terms: list[dict[str, Any]] = []

    for term, doc_count in term_doc_count.items():
        sources_with_term = term_source_count.get(term, set())
        n_sources = len(sources_with_term)
        total_freq = term_total_count.get(term, 1)

        # IDF calculation
        idf = math.log((n_docs + 1) / (doc_count + 1)) + 1.0

        # Multi-source and multi-window multiplier
        source_bonus = 1.0 + 0.5 * (n_sources - 1)
        bigram_bonus = 1.25 if " " in term else 1.0

        score = total_freq * idf * source_bonus * bigram_bonus

        # Select up to 3 diverse snippets
        cand_docs = term_snippets.get(term, [])
        selected_snippets: list[dict[str, Any]] = []
        seen_channels: set[str] = set()

        # First pass: distinct channels
        for d in cand_docs:
            ch = d["channel"] or d["source_id"] or "Unknown"
            if ch not in seen_channels and len(selected_snippets) < 3:
                seen_channels.add(ch)
                clean_snippet = _extract_snippet_window(d["text"], term, max_chars=140)
                mm_ss = _format_time(d["time"])
                selected_snippets.append({
                    "channel": ch,
                    "time": mm_ss,
                    "text": clean_snippet,
                    "formatted": f"{ch} ({mm_ss}): {clean_snippet}",
                })

        # Second pass: fill up to 3 if needed
        for d in cand_docs:
            if len(selected_snippets) >= 3:
                break
            ch = d["channel"] or d["source_id"] or "Unknown"
            clean_snippet = _extract_snippet_window(d["text"], term, max_chars=140)
            mm_ss = _format_time(d["time"])
            formatted_entry = {
                "channel": ch,
                "time": mm_ss,
                "text": clean_snippet,
                "formatted": f"{ch} ({mm_ss}): {clean_snippet}",
            }
            if formatted_entry not in selected_snippets:
                selected_snippets.append(formatted_entry)

        # Top co-occurring terms
        co_list = [
            t for t, _ in sorted(
                cooccurrence.get(term, {}).items(),
                key=lambda item: item[1],
                reverse=True,
            )[:8]
        ]

        scored_terms.append({
            "term": term,
            "score": round(score, 3),
            "sources": n_sources,
            "snippets": selected_snippets,
            "co_occurring": co_list,
        })

    scored_terms.sort(key=lambda x: (x["score"], x["sources"]), reverse=True)
    top_terms = scored_terms[:max(1, max_terms)]

    return {
        "game": game,
        "stats": stats,
        "existing_titles": existing_titles,
        "terms": top_terms,
    }


def evaluate_topic(
    config: Config,
    game: str,
    title: str,
    keywords: list[str],
) -> TopicCandidate:
    """Evaluate a proposed topic against the library index and topics ledger.

    Searches the index for the title and each keyword, keeps best distinct spans
    up to topics.max_evidence (default 12), ensures multi-source diversity to avoid
    evidence starvation across sources, and sets already_made/similar_to from the ledger.
    """
    clean_title = str(title or "").strip()
    topic_id = _slugify(clean_title)
    if isinstance(keywords, str):
        clean_kw = [k.strip() for k in keywords.split(",") if k.strip()]
    else:
        clean_kw = [str(k).strip() for k in (keywords or []) if str(k).strip()]

    # Check ledger for existing/similar topics
    ledger = TopicLedger(config)
    similar_entry = ledger.find_similar(clean_title, clean_kw, game=game)
    already_made = similar_entry is not None
    similar_to = str(similar_entry.get("topic_id", "")) if similar_entry else ""

    min_evidence = int(config.get("topics.min_evidence", 6))
    min_sources = int(config.get("topics.min_sources", 2))
    max_evidence = int(config.get("topics.max_evidence", 12))
    index = LibraryIndex(config)

    # Search queries: title and each keyword
    queries = [clean_title] + clean_kw
    candidate_hits: list[Any] = []

    for q in queries:
        if not q:
            continue
        hits = index.search(q, game=game or None, limit=max_evidence * 2)
        candidate_hits.extend(hits)

    # Sort candidate hits by score descending
    candidate_hits.sort(key=lambda h: float(h.score), reverse=True)

    # Group non-overlapping hits per source
    from collections import defaultdict
    source_hits_map: dict[str, list[Any]] = defaultdict(list)
    for hit in candidate_hits:
        src_id = hit.source_id
        # Check overlap with already accepted hits for this source
        overlaps = False
        for existing in source_hits_map[src_id]:
            if not (hit.end <= existing.start or hit.start >= existing.end):
                overlaps = True
                break
        if not overlaps:
            source_hits_map[src_id].append(hit)

    distinct_evidence: list[EvidenceRef] = []

    # Round 1 (Source Diversity): Ensure each source with hits gets at least 1 top hit
    # so a single source with many hits never starves other sources of evidence slots
    for src_id, src_hits in list(source_hits_map.items()):
        if src_hits and len(distinct_evidence) < max_evidence:
            hit = src_hits.pop(0)
            distinct_evidence.append(
                EvidenceRef(
                    source_id=hit.source_id,
                    url=hit.url,
                    channel=hit.channel,
                    kind=hit.kind,
                    start=round(float(hit.start), 2),
                    end=round(float(hit.end), 2),
                    snippet=hit.snippet,
                    heat=round(float(hit.heat), 4),
                    score=round(float(hit.score), 4),
                )
            )

    # Round 2: Fill remaining quota with highest-scoring hits across all sources
    remaining_hits = sorted(
        [h for src_hits in source_hits_map.values() for h in src_hits],
        key=lambda h: float(h.score),
        reverse=True,
    )
    for hit in remaining_hits:
        if len(distinct_evidence) >= max_evidence:
            break
        distinct_evidence.append(
            EvidenceRef(
                source_id=hit.source_id,
                url=hit.url,
                channel=hit.channel,
                kind=hit.kind,
                start=round(float(hit.start), 2),
                end=round(float(hit.end), 2),
                snippet=hit.snippet,
                heat=round(float(hit.heat), 4),
                score=round(float(hit.score), 4),
            )
        )

    # Sort final distinct evidence by score descending
    distinct_evidence.sort(key=lambda e: float(e.score), reverse=True)

    source_count = len({e.source_id for e in distinct_evidence if e.source_id})
    channel_count = len({e.channel for e in distinct_evidence if e.channel})

    return TopicCandidate(
        topic_id=topic_id,
        title=clean_title,
        game=game,
        description=f"Curiosity-driven topic with {len(distinct_evidence)} evidence clips from {source_count} sources.",
        keywords=clean_kw,
        evidence=distinct_evidence,
        source_count=source_count,
        channel_count=channel_count,
        already_made=already_made,
        similar_to=similar_to,
        min_evidence=min_evidence,
        min_sources=min_sources,
    )


def _fallback_term_clusters(terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group top salient terms into curiosity-driven proposal clusters when Ollama is unavailable.

    Groups terms by co-occurrence when co-occurring metadata is available; otherwise groups
    consecutive salient terms.
    """
    if not terms:
        return []

    clusters: list[dict[str, Any]] = []
    unclustered = list(terms)

    while unclustered:
        seed = unclustered.pop(0)
        seed_term = seed["term"]
        cluster_terms = [seed_term]

        # Check for co-occurring terms among remaining unclustered terms
        co_candidates = seed.get("co_occurring") or []
        for co_t in co_candidates:
            if len(cluster_terms) >= 4:
                break
            match = next((t for t in unclustered if t["term"] == co_t), None)
            if match:
                cluster_terms.append(co_t)
                unclustered.remove(match)

        # Fill up to cluster size with remaining unclustered if needed
        while len(cluster_terms) < min(4, len(cluster_terms) + len(unclustered)):
            cluster_terms.append(unclustered.pop(0)["term"])

        lead_term = cluster_terms[0].title()
        second_term = cluster_terms[1].title() if len(cluster_terms) > 1 else ""
        if second_term:
            title = f"Every Hidden Detail In {lead_term} & {second_term}"
        else:
            title = f"The Complete Investigation of {lead_term}"

        clusters.append({
            "title": title,
            "keywords": cluster_terms,
            "angle": f"In-depth breakdown of key details around {', '.join(cluster_terms)}.",
        })

    return clusters


def suggest_topics_local(
    config: Config,
    game: str,
    *,
    limit: int = 10,
) -> list[TopicCandidate]:
    """Suggest viable topic video ideas using local Ollama or fallback term clusters.

    Returns viable candidates sorted by evidence strength, with already-made topics last.
    """
    material = topic_material(config, game, max_terms=40)
    terms = material.get("terms", [])
    if not terms:
        return []

    prompt_file = Path(__file__).parent.parent / "prompts" / "topic_discovery.txt"
    template = prompt_file.read_text(encoding="utf-8") if prompt_file.exists() else ""

    terms_summary_lines: list[str] = []
    for t_info in terms[:25]:
        snippets_txt = " | ".join(s["formatted"] for s in t_info["snippets"])
        terms_summary_lines.append(f"- Term: '{t_info['term']}' (Sources: {t_info['sources']}) -> {snippets_txt}")
    terms_summary = "\n".join(terms_summary_lines)

    existing_titles_txt = "\n".join(f"- {t}" for t in material.get("existing_titles", [])) or "None"

    formatted_prompt = (
        template.replace("{game}", str(game))
        .replace("{total_sources}", str(material["stats"]["sources"]))
        .replace("{total_channels}", str(material["stats"]["channels"]))
        .replace("{kinds}", ", ".join(material["stats"]["kinds"]) or "stream")
        .replace("{total_hours:.1f}", f"{material['stats']['total_hours']:.1f}")
        .replace("{total_hours}", f"{material['stats']['total_hours']:.1f}")
        .replace("{existing_titles}", existing_titles_txt)
        .replace("{terms_summary}", terms_summary)
        .replace("{limit}", str(max(5, limit)))
    ) if template else ""

    items: list[dict[str, Any]] = []

    # Attempt local Ollama inference
    if formatted_prompt:
        try:
            import ollama
            base_url = str(config.get("llm.base_url", "http://localhost:11434") or "http://localhost:11434")
            timeout = float(config.get("llm.timeout", 30.0))
            model = str(config.get("llm.model", "qwen3:8b") or "qwen3:8b")
            client = ollama.Client(host=base_url, timeout=timeout)
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": formatted_prompt}],
                options={"temperature": 0.4},
            )
            raw_text = str(response.get("message", {}).get("content", "")).strip()

            parsed = _extract_json(raw_text)
            if isinstance(parsed, dict):
                if "topics" in parsed and isinstance(parsed["topics"], list):
                    items = parsed["topics"]
                elif "candidates" in parsed and isinstance(parsed["candidates"], list):
                    items = parsed["candidates"]
                elif "title" in parsed:
                    items = [parsed]
            elif isinstance(parsed, list):
                items = parsed
        except Exception as exc:
            logger.info("Ollama topic discovery not available or invalid response: %s", exc)
            items = []

    if not items:
        items = _fallback_term_clusters(terms)

    # Evaluate each proposal
    evaluated_candidates: list[TopicCandidate] = []
    seen_slugs: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        slug = _slugify(title)
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        raw_kw = item.get("keywords") or []
        if isinstance(raw_kw, str):
            kw_list = [k.strip() for k in raw_kw.split(",") if k.strip()]
        else:
            kw_list = [str(k).strip() for k in raw_kw if str(k).strip()]
        candidate = evaluate_topic(config, game, title, kw_list)
        if item.get("angle"):
            candidate.description = str(item["angle"])

        evaluated_candidates.append(candidate)

    # Filter to viable candidates
    viable = [c for c in evaluated_candidates if c.is_viable]

    # Sort: already-made last, then by evidence count and total score descending
    viable.sort(
        key=lambda c: (
            1 if c.already_made else 0,
            -len(c.evidence),
            -sum(e.score for e in c.evidence),
        )
    )

    if limit > 0:
        return viable[:limit]
    return viable

    # Filter to viable candidates
    viable = [c for c in evaluated_candidates if c.is_viable]

    # Sort: already-made last, then by evidence count and total score descending
    viable.sort(
        key=lambda c: (
            1 if c.already_made else 0,
            -len(c.evidence),
            -sum(e.score for e in c.evidence),
        )
    )

    if limit > 0:
        return viable[:limit]
    return viable
