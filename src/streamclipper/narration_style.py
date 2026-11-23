"""Shared narration style guide + quality checks (Ollama prompt and MCP host-LLM guide).

One source of truth for the energetic, interactive "gamer friend reacting live" voice:
``narration_guide()`` builds the writing brief, ``check_narration()`` flags lines that are
generic, ungrounded, non-interactive or repeat recent clips.
"""

from __future__ import annotations

import re
from typing import Any

from .language import HINGLISH, language_instruction

# ---------------------------------------------------------------------------
# Thresholds (module constants)
# ---------------------------------------------------------------------------
DEFAULT_MAX_WORDS_PER_SENTENCE = 14  # prompt asks for 12, the check allows a little slack
AVOID_TOKEN_OVERLAP = 0.8  # >= this share of tokens shared with a recent line = repeat
AVOID_MIN_TOKENS = 3  # ignore overlap on very short fragments
OPENING_WORDS = 3  # intro/outro must not share this many leading words
MIN_CONTENT_WORD_LEN = 4  # grounding: letters in a "content word"
MIN_DEVANAGARI_WORD_LEN = 3  # Devanagari words are shorter
DEVANAGARI_TRANSCRIPT_RATIO = 0.3  # transcript mostly Devanagari -> grounding check skipped

STYLE_RULES = """\
VOICE: you are a hyped gamer friend reacting LIVE next to the viewer, not a narrator. High energy, funny, a bit unhinged, always rooting for the streamer (or roasting them with love).
- Talk DIRECTLY to the viewer: "you", "bro", "look at this", "check this out".
- Use reaction energy: short exclamations, rhetorical questions ("Did he really just...?"), hype build-up, mock disbelief.
- Use gaming slang where it sounds natural: clutch, cooked, sent it, no way, GG, "bro really said...". Do not force it.
- SPECIFICS BEAT ENERGY: every line must hook onto a concrete detail from the TRANSCRIPT EXCERPT or CHAT (what the streamer said or did, an object, a chat message). Energy never replaces specifics. Never invent events.
- Every sentence is 12 words or fewer. Short, punchy, spoken out loud.
- INTRO (narration_intro, 1-2 sentences): hook the viewer into what is about to happen WITHOUT spoiling the payoff.
- CALLOUT (callout, 4-6 words): hype words said right before the payoff, e.g. "Wait wait wait, watch the bridge!".
- OUTRO (narration_outro, 1-2 sentences): YOUR OWN reaction, verdict or question to the viewer that invites comments, e.g. "Would you have bailed out there? Be honest." NEVER a summary of what happened. NEVER ask for subscribe / follow / like.
- VARY your openings: no two clips may start the same way."""

# moment_type -> energetic, interactive tone line (badge examples kept for the badge field).
TONE_BY_MOMENT_TYPE: dict[str, str] = {
    "funny": "funny: crack up WITH the viewer. Set the joke up with a smirk, roast the streamer lovingly, land a punchy outro ('Be honest, would you have laughed too?'). Badge e.g. \"💀 Instant Karma\".",
    "fail": "fail: pure disbelief and mock sympathy. Build it up like you saw the disaster coming, then react to the wreckage and ask the viewer if they have done worse. Badge e.g. \"💀 Instant Karma\".",
    "graphics_detail": "graphics_detail: nerd out out loud. 'Look at this!' pointing at the exact detail, then ask the viewer if they noticed it. Badge e.g. \"🎨 Insane Detail\".",
    "hype": "hype: scream-level excitement with a build-up. Stakes up, 'no way', then a reaction outro that asks the viewer what they would have done. Badge e.g. \"🔥 Clutch Escape\".",
    "clutch": "clutch: edge-of-your-seat stakes ('one shot, no backup'), then pure hype at the result and a question like 'Could you pull that off?'. Badge e.g. \"🔥 Clutch Escape\".",
    "chat_chaos": "chat_chaos: stream vs chat banter. Quote what chat is yelling, set up the streamer ignoring (or obeying) it, then ask the viewer whose side they are on. Badge e.g. \"💬 Chat Trolled Him\".",
    "lore_secret": "lore_secret: whisper-hype like you just found a secret. 'Wait, did you know this?', tease the hidden thing, then ask if the viewer already knew. Badge e.g. \"🧠 Hidden Detail\".",
    "analysis": "analysis: still a friend, not a lecturer. One sharp 'here is the thing' observation tied to a detail, then ask the viewer for their take. Badge e.g. \"🧠 Hidden Detail\".",
    "reaction": "reaction: set up what the streamer is staring at, then riff on their face and voice like you are sitting beside them. Badge e.g. \"😳 Pure Shock\".",
}

# Style examples only. The model must write NEW lines about the actual clip.
EXAMPLES_EN: tuple[dict[str, str], ...] = (
    {
        "moment_type": "clutch",
        "narration_intro": "Cops everywhere, one bridge, zero plan. Watch this.",
        "callout": "Wait wait wait, the bridge!",
        "narration_outro": "Would you have bailed out there? Be honest.",
    },
    {
        "moment_type": "fail",
        "narration_intro": "He said 'trust me' about the jet ski. Bro, no.",
        "callout": "Oh no, he's going in!",
        "narration_outro": "Tell me you have never done something that dumb. Liar.",
    },
    {
        "moment_type": "chat_chaos",
        "narration_intro": "Chat spammed 'don't do it'. He did not listen.",
        "callout": "Chat was right, look!",
        "narration_outro": "Whose side are you on, chat or the streamer?",
    },
)

EXAMPLES_HINGLISH: tuple[dict[str, str], ...] = (
    {
        "moment_type": "hype",
        "narration_intro": "Bhai ne police car leke bridge pe full send kar diya!",
        "callout": "Ruk ruk ruk, bridge dekh!",
        "narration_outro": "Tum hote toh brake maarte ya send karte? Comment karo.",
    },
    {
        "moment_type": "chat_chaos",
        "narration_intro": "Chat bol raha tha mat kar... aur bhai ne kar diya!",
        "callout": "Chat sahi tha, dekho!",
        "narration_outro": "Tum kiski side ho, chat ki ya streamer ki?",
    },
    {
        "moment_type": "funny",
        "narration_intro": "Helicopter ko dekh ke bhai bola 'main sambhal lunga'. Lol.",
        "callout": "Arre arre arre, ye kya!",
        "narration_outro": "Bhai ka confidence dekha? Aisa dost tumhare paas bhi hai na?",
    },
)

# Back-compat / reviewer-friendly flat view.
EXAMPLES: dict[str, tuple[dict[str, str], ...]] = {
    "english": EXAMPLES_EN,
    "hinglish": EXAMPLES_HINGLISH,
}

# D4: generic phrases the narrative writer must never produce (lower-case substring match).
BANNED_PHRASES: tuple[str, ...] = (
    "the chat went wild",
    "chat went completely wild",
    "you won't believe",
    "watch what happens",
    "subscribe for more",
    "over a million people",
    "insane moment",
    "most viral",
    "pure comedy right here",
    "this is the single",
    "launch stream",
    "like and subscribe",
    "smash that like",
    "don't forget to",
    "in this clip",
    "in today's video",
    "let's take a look",
    "what happens next",
    "you have to see",
    "absolutely insane",
    "crazy moment",
)

_STOPWORDS = frozenset(
    """
    this that these those with have has had from they them their there then than what when where which while
    will would could should about into over under just like very much more most some such only also even
    your yours yourself were been being does doing done dont cant wont isnt arent wasnt here look watch wait
    clip stream video game gaming bhai yaar kuch kiya kiye karo kare hain hai hoga hota hote thaa
    aapko tumhe tumne mera mere meri apna apne uska uski unka yeh woh jaise lekin kyunki phir abhi bahut
    """.split()
)

_TOKEN_RE = re.compile(r"(?:[^\W\d_]|[ऀ-ॿ])+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?…।\n]+")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_NOT_SUBSCRIBE_RE = re.compile(
    r"\bsubscrib\w*|\bfollow\s+(?:me|us|for|along|the\s+channel|my)\b|\bsub\s+(?:for|to)\b"
    r"|\b(?:smash|hit|drop|leave|give|press)\b.{0,14}\b(?:likes?|button)\b"
    r"|\blike\s+(?:and|&|this|the)\b",
    re.IGNORECASE,
)
_VIEWER_WORDS = frozenset(
    "you your youre you're yours tum tumhe tumhara tumhari tumne aap aapka aapko bhai bro comment guys".split()
)


def _norm(text: Any) -> str:
    return str(text or "").replace("’", "'").replace("‘", "'").strip()


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(_norm(text))]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(_norm(text)) if s.strip()]


def _content_words(text: str) -> set[str]:
    out: set[str] = set()
    for tok in _tokens(text):
        dev = bool(_DEVANAGARI_RE.search(tok))
        if len(tok) >= (MIN_DEVANAGARI_WORD_LEN if dev else MIN_CONTENT_WORD_LEN) and tok not in _STOPWORDS:
            out.add(tok)
    return out


def _overlap(a: list[str], b: list[str]) -> float:
    if min(len(a), len(b)) < AVOID_MIN_TOKENS:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, min(len(sa), len(sb)))


def narration_guide(
    language: str,
    hook_style: str = "",
    avoid_phrases: list[str] | None = None,
) -> str:
    """Assemble the writing brief: rules, tone per moment_type, examples, banned + avoid lists."""
    lang = str(language or "").strip().lower()
    parts: list[str] = [STYLE_RULES, "", "TONE PER moment_type:"]
    parts += [f"- {line}" for line in TONE_BY_MOMENT_TYPE.values()]

    examples = list(EXAMPLES_EN)
    if lang == HINGLISH:
        examples += list(EXAMPLES_HINGLISH)
    parts += [
        "",
        "STYLE EXAMPLES (for energy and structure ONLY. Do NOT copy them; write new lines about THIS clip's real details):",
    ]
    for ex in examples:
        parts.append(
            f'- [{ex["moment_type"]}] intro: "{ex["narration_intro"]}" | '
            f'callout: "{ex["callout"]}" | outro: "{ex["narration_outro"]}"'
        )

    instruction = language_instruction(lang)
    if instruction:
        parts += ["", instruction]
    if hook_style and str(hook_style).strip().lower() != "any":
        parts += ["", f"Opening style for this clip: {str(hook_style).strip()}."]
    parts += [
        "",
        "BANNED generic phrases (never use): " + ", ".join(f'"{p}"' for p in BANNED_PHRASES) + ".",
    ]
    avoid = [str(p).strip() for p in (avoid_phrases or []) if p and str(p).strip()]
    if avoid:
        parts.append(
            "Also do NOT repeat these lines/openings used in recent clips: "
            + "; ".join(f'"{p}"' for p in avoid)
            + "."
        )
    return "\n".join(parts)


def check_narration(
    fields: dict[str, Any],
    *,
    transcript_excerpt: str = "",
    chat_samples: list[str] | None = None,
    avoid_phrases: list[str] | None = None,
    max_words_per_sentence: int = DEFAULT_MAX_WORDS_PER_SENTENCE,
    require_grounding: bool = False,
) -> list[str]:
    """Return human-readable problems with the narration lines (empty list = OK).

    *fields* may contain ``narration_intro``, ``narration_outro`` and ``callout``; empty or
    missing fields are skipped (e.g. when a voice file replaces the text).
    """
    intro = _norm(fields.get("narration_intro"))
    outro = _norm(fields.get("narration_outro"))
    callout = _norm(fields.get("callout"))
    named = [("narration_intro", intro), ("callout", callout), ("narration_outro", outro)]
    named = [(n, t) for n, t in named if t]
    problems: list[str] = []
    if not named:
        return problems

    # 1. banned generic phrases
    for name, text in named:
        low = text.lower()
        for phrase in BANNED_PHRASES:
            if phrase in low:
                problems.append(f'{name} uses the banned generic phrase "{phrase}"')

    # 2. repeats a recent outro/opening
    avoid = [_norm(p) for p in (avoid_phrases or []) if _norm(p)]
    for name, text in named:
        cand_texts = [text] + _sentences(text)
        flagged: str | None = None
        for a in avoid:
            a_low = a.lower()
            a_texts = [a] + _sentences(a)
            for ct in cand_texts:
                ct_low = ct.lower()
                if len(a_low) >= 6 and a_low in ct_low:
                    flagged = a
                elif len(ct_low) >= 12 and ct_low in a_low:
                    flagged = a
                else:
                    ctoks = _tokens(ct)
                    if any(_overlap(ctoks, _tokens(at)) >= AVOID_TOKEN_OVERLAP for at in a_texts):
                        flagged = a
                if flagged:
                    break
            if flagged:
                break
        if flagged:
            problems.append(f'{name} repeats a recent clip\'s line ("{flagged[:60]}"); write something new')

    # 3. sentence length
    for name, text in named:
        for sent in _sentences(text):
            n = len(_tokens(sent))
            if n > max_words_per_sentence:
                problems.append(
                    f'{name} has a sentence of {n} words (max {max_words_per_sentence}): "{sent[:50]}"'
                )

    # 4. outro asks for subscribe / follow / like
    if outro and _NOT_SUBSCRIBE_RE.search(outro):
        problems.append("narration_outro asks to subscribe/follow/like; ask the viewer a question instead")

    # 5. intro and outro open the same way
    if intro and outro:
        a, b = _tokens(intro)[:OPENING_WORDS], _tokens(outro)[:OPENING_WORDS]
        if len(a) == OPENING_WORDS and a == b:
            problems.append(f'intro and outro start with the same {OPENING_WORDS} words')

    # 6. grounding
    source = " ".join([_norm(transcript_excerpt)] + [_norm(c) for c in (chat_samples or [])]).strip()
    if not source:
        # No transcript or chat to ground narration against
        if require_grounding and (intro or outro or callout):
            problems.append(
                "UNGROUNDED: narration lines provided but no transcript excerpt or chat "
                "samples are available to ground them. Run transcribe_audio first or use mode: raw."
            )
    else:
        dev_chars = len(_DEVANAGARI_RE.findall(source))
        letters = max(1, len(re.findall(r"\w", source)))
        if dev_chars / letters < DEVANAGARI_TRANSCRIPT_RATIO:
            src_words = _content_words(source)
            line_words: set[str] = set()
            for _, text in named:
                line_words |= _content_words(text)
            if not (src_words & line_words):
                problems.append(
                    "NOT GROUNDED: no line mentions a concrete detail (a word the streamer said, an "
                    "object, a chat message) from the transcript excerpt or chat"
                )

    # 7. interactive outro
    if outro:
        toks = set(_tokens(outro))
        if "?" not in outro and not (toks & _VIEWER_WORDS):
            problems.append(
                "NOT INTERACTIVE: narration_outro has no viewer-directed element (ask a question or "
                "address the viewer with you/bro/comment)"
            )

    return problems

