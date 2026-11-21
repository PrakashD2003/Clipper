"""Tests for topic video planner, validation, approval, and MCP/CLI tools (T4)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import yaml
from click.testing import CliRunner

from streamclipper.cli import cli
from streamclipper.config import Config
from streamclipper.library.index import LibraryIndex
from streamclipper.library.models import LibrarySource
from streamclipper.mcp_server.server import (
    approve_topic_plan,
    get_topic_plan,
    start_topic_plan,
    submit_topic_plan,
    update_topic_plan,
)
from streamclipper.models import (
    PipelineState,
    PipelineStatus,
    Platform,
    StreamInfo,
    TranscriptSegment,
    TranscriptionResult,
)
from streamclipper.topics.discover import evaluate_topic
from streamclipper.topics.ledger import TopicLedger
from streamclipper.topics.models import (
    EvidenceClip,
    ShortPlan,
    TopicCandidate,
    TopicPlan,
    TopicPoint,
)
from streamclipper.topics.planner import (
    apply_merge_patch,
    choose_shape,
    format_plan_outline,
    load_plan,
    normalize_plan_clips,
    plan_dir,
    plan_topic_local,
    planning_material,
    save_plan,
)
from streamclipper.topics.validate import validate_plan


def _make_config(tmp_path: Path, **extra: Any) -> Config:
    """Create a test Config pointing cache, output, and temp to tmp_path."""
    cache_dir = tmp_path / ".cache"
    temp_dir = tmp_path / "temp"
    output_dir = tmp_path / "output"
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "general": {
            "cache_dir": str(cache_dir),
            "temp_dir": str(temp_dir),
            "output_dir": str(output_dir),
        },
        "library": {
            "db_path": str(cache_dir / "library.sqlite"),
            "window_seconds": 30,
            "window_step_seconds": 20,
            "max_chat_chars_per_window": 600,
            "heat_boost": 0.3,
            "frames": {"note_weight": 0.8},
        },
        "topics": {
            "min_evidence": 6,
            "min_sources": 2,
            "max_evidence": 12,
            "similarity_threshold": 0.5,
            "long_min_seconds": 480,
            "long_max_seconds": 600,
            "min_narration_ratio": 0.40,
            "max_clip_seconds": 20,
            "max_official_clip_seconds": 10,
            "short_min_seconds": 35,
            "short_max_seconds": 60,
        },
    }
    data.update(extra)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    return Config(user_config_path=cfg_file)


def _build_state(
    url: str,
    channel: str,
    game: str,
    duration: int,
    segments: list[TranscriptSegment],
) -> PipelineState:
    full_text = " ".join(s.text for s in segments)
    video_id = url.split("v=")[-1] if "v=" in url else "vid"
    return PipelineState(
        stream_info=StreamInfo(
            url=url,
            title=f"Stream from {channel}",
            platform=Platform.YOUTUBE,
            streamer=channel,
            game=game,
            duration_seconds=duration,
            video_id=video_id,
        ),
        status=PipelineStatus.SCORING,
        transcript=TranscriptionResult(
            segments=segments,
            full_text=full_text,
            language="en",
        ),
        loudness_spikes=[],
        heatmap_spikes=[],
        analysis_complete=True,
    )


@pytest.fixture
def sample_library(tmp_path: Path) -> tuple[Config, LibraryIndex]:
    """Populate a test SQLite library with stream, gameplay, and official sources."""
    cfg = _make_config(tmp_path)
    index = LibraryIndex(cfg)

    # Source 1: Streamer AlphaGamer (English)
    segs1 = [
        TranscriptSegment(start=10.0, end=25.0, text="Cargo ship dock secret vault entrance found right here"),
        TranscriptSegment(start=150.0, end=165.0, text="Security guards patrol the warehouse waterfront area"),
        TranscriptSegment(start=300.0, end=315.0, text="Hidden container opened with rare sports car inside"),
        TranscriptSegment(start=450.0, end=465.0, text="Underground escape tunnel leads directly to the ocean"),
        TranscriptSegment(start=600.0, end=615.0, text="Police boat chase around the harbor pier"),
    ]
    s1 = _build_state("https://youtube.com/watch?v=src1", "AlphaGamer", "GTA6", 800, segs1)
    index.index_state(s1, kind="stream")

    # Source 2: Streamer DesiPlayer (Hinglish)
    segs2 = [
        TranscriptSegment(start=20.0, end=35.0, text="Cargo ship crane control panel hack kar diya"),
        TranscriptSegment(start=180.0, end=195.0, text="Warehouse basement door unlock ho gaya secret location"),
        TranscriptSegment(start=350.0, end=365.0, text="Police helicopter patrol kar rahi hai docks ke upar"),
        TranscriptSegment(start=500.0, end=515.0, text="Speedboat getaway route near the lighthouse"),
    ]
    s2 = _build_state("https://youtube.com/watch?v=src2", "DesiPlayer", "GTA6", 800, segs2)
    index.index_state(s2, kind="stream")

    # Source 3: Gameplay video (no-commentary)
    segs3 = [
        TranscriptSegment(start=10.0, end=25.0, text="Harbor overview shipping facility"),
        TranscriptSegment(start=500.0, end=515.0, text="Lighthouse ocean coast boundary"),
    ]
    s3 = _build_state("https://youtube.com/watch?v=src3", "SilentWalker", "GTA6", 800, segs3)
    index.index_state(s3, kind="gameplay")

    # Source 4: Official Rockstar Reveal Trailer
    segs4 = [
        TranscriptSegment(start=0.0, end=20.0, text="Grand Theft Auto VI Official Trailer Rockstar Games"),
    ]
    s4 = _build_state("https://youtube.com/watch?v=src_official", "RockstarGames", "GTA6", 90, segs4)
    index.index_state(s4, kind="official")

    return cfg, index


def _build_valid_plan(cfg: Config, index: LibraryIndex) -> TopicPlan:
    """Construct a fully valid 5-point numbered TopicPlan that satisfies every contract rule."""
    points = [
        TopicPoint(
            index=1,
            heading="#1 The Secret Vault",
            narration=(
                "Check out this discovery bro. "
                "The cargo ship dock hides a secret entrance. "
                "AlphaGamer finds the hidden door behind the container. "
                "Nobody expected this secret vault here. "
                "Look closely at the rusty lock mechanism. "
                "The steel door swings open slowly right now. "
                "Check out the computer terminal glowing inside. "
                "This discovery completely changes the harbor escape route. "
                "Every security guard was looking the other way. "
                "Now we see what was inside the vault. "
                "Notice the rare sports car parked in the corner. "
                "That explains why the cameras were watching this spot. "
                "You can drive right out through the loading bay. "
                "The secret passage connects directly to the lower level. "
                "Stream chat was spamming GG when the door opened. "
                "Nobody had found this room before this stream. "
                "This secret vault is the biggest find in Vice City. "
                "Make sure you remember this location for your next run. "
                "The loot inside respawns every twenty four hours in game. "
                "You can farm rare weapon attachments here effortlessly. "
                "Check the weapon crate hidden behind the generator. "
                "It gives you free body armor every time."
            ),
            evidence=[
                EvidenceClip(
                    source_id="https://youtube.com/watch?v=src1",
                    start=10.0,
                    end=28.0,
                    role="evidence",
                    reason="Cargo ship dock secret vault entrance found",
                )
            ],
            use_in=["long", "teaser"],
        ),
        TopicPoint(
            index=2,
            heading="#2 Warehouse Security Patrol",
            narration=(
                "Security guards patrol the warehouse waterfront area day and night. "
                "DesiPlayer spotted the exact guard rotation timing. "
                "Look at the sniper guarding the rooftop ledge. "
                "The warehouse basement door unlock was completely silent. "
                "They never saw him coming through the dark shadows. "
                "The entire dock perimeter is covered by motion sensors. "
                "Notice how the guard turns every thirty seconds. "
                "That gave him the exact window to slip past. "
                "He hacked the security gate without triggering any alarms. "
                "This reveals a huge blind spot in harbor security. "
                "Watch the guard walk right past his hiding spot. "
                "Timing that patrol route was pure genius bro. "
                "The basement hallway leads under the main shipping office. "
                "You can bypass the heavy metal gates completely this way. "
                "He grabbed the keycard off the desk unnoticed. "
                "This gives you access to the entire warehouse interior. "
                "Always check behind the crates before moving forward here. "
                "The guards never look behind the shipping pallets. "
                "You can disable the warehouse power grid from this console. "
                "Cutting the lights leaves the entire squad stumbling around. "
                "Head straight up the metal stairs to grab the blueprints. "
                "They reveal the shipping schedule for every incoming container."
            ),
            evidence=[
                EvidenceClip(
                    source_id="https://youtube.com/watch?v=src2",
                    start=180.0,
                    end=198.0,
                    role="evidence",
                    reason="Warehouse basement door unlock secret location",
                )
            ],
            use_in=["long", "short_a"],
        ),
        TopicPoint(
            index=3,
            heading="#3 Crane Control Hack",
            narration=(
                "The cargo ship crane moves silently across the shipyard rails. "
                "He reached the crane control panel in twenty seconds flat. "
                "Watch the heavy shipping container lift into the foggy sky. "
                "The crane drops the massive cargo onto the upper pier. "
                "This opens a completely new vantage point for players. "
                "From up here the whole harbor perimeter is visible. "
                "The high camera view shows every patrol boat below. "
                "Notice the hidden container resting on the upper deck. "
                "Nobody had climbed this industrial crane before this attempt. "
                "The steel ladder leads straight to the operator cab. "
                "He used the long crane arm as a bridge across. "
                "That jump between containers was totally insane bro. "
                "You can drop right onto the roof from here. "
                "It skips the entire security checkpoint down on the ground. "
                "The crane panel also unlocks the harbor floodlights. "
                "Turning off the lights blinds the snipers instantly. "
                "This is one of the most clever shortcuts discovered. "
                "Keep this crane trick in your playbook. "
                "You can swing containers into enemy vehicles parked below. "
                "The physics engine handles the collision destruction beautifully. "
                "Climbing down the rear ladder brings you to the harbor docks. "
                "You can steal the harbormaster boat with zero stars."
            ),
            evidence=[
                EvidenceClip(
                    source_id="https://youtube.com/watch?v=src2",
                    start=20.0,
                    end=38.0,
                    role="evidence",
                    reason="Cargo ship crane control panel hack",
                )
            ],
            use_in=["long", "short_b"],
        ),
        TopicPoint(
            index=4,
            heading="#4 Underground Tunnel Escape",
            narration=(
                "An underground escape tunnel leads directly out to the ocean. "
                "AlphaGamer found the submerged water tunnel exit by accident. "
                "The secret passage is hidden beneath the old pier pilings. "
                "Look at the water currents rushing around the tunnel entrance. "
                "This passage connects directly to the open shipping channels. "
                "You can swim through without alerting the harbor patrol. "
                "The tunnel cuts through the entire dock foundation wall. "
                "Notice the air pockets spaced along the concrete ceiling. "
                "It completely bypasses all the police roadblock checkpoints on land. "
                "The water exit emerges safely behind the lighthouse rocks. "
                "Watch him dive straight into the dark water passage. "
                "This is the ultimate getaway route from the docks. "
                "No police cars can follow you into this submerged tunnel. "
                "You pop out into deep water with zero heat. "
                "Grab a scuba suit before attempting this dive bro. "
                "The current pushes you right out to the open bay. "
                "It makes escaping five star wanted levels totally effortless. "
                "This tunnel is an absolute game changer. "
                "Keep an eye out for underwater weapon crates down here. "
                "You will find suppressed rifles tucked in the rocks. "
                "Follow the underwater pipeline toward the deep channel buoys. "
                "A getaway jet ski is moored right past the reef."
            ),
            evidence=[
                EvidenceClip(
                    source_id="https://youtube.com/watch?v=src1",
                    start=450.0,
                    end=468.0,
                    role="evidence",
                    reason="Underground escape tunnel leads directly to ocean",
                )
            ],
            use_in=["long"],
        ),
        TopicPoint(
            index=5,
            heading="#5 Harbor Boat Chase",
            narration=(
                "A wild police boat chase erupts around the harbor pier. "
                "The speedboat getaway route near the lighthouse opens up now. "
                "Police boats patrol aggressively near the docks boundary line. "
                "Watch the sharp turn he takes around the red buoy. "
                "The heavy harbor police cruisers could not keep up here. "
                "Massive ocean waves crash against the rocky coastline cliffs. "
                "He threaded the needle cleanly between two huge cargo ships. "
                "Notice how the patrol boats completely lose radar tracking. "
                "The twin speedboat engines hit maximum speed right now. "
                "He vanished smoothly into the thick open ocean fog. "
                "Watch the searchlights sweep the empty water far behind him. "
                "That clean escape wraps up the entire dock investigation. "
                "The ocean boundary route gives you complete freedom to vanish. "
                "Never try to escape on foot when boats are available. "
                "The harbor police simply cannot match this top speed. "
                "That marks the end of our dock secrets breakdown. "
                "Make sure to test this route on your next escape. "
                "You will leave the cops in your wake every time. "
                "Speedboat handling around these narrow docks feels incredible. "
                "Practice turning in the tight canals before the cops arrive. "
                "The open sea escape resets your wanted status in seconds. "
                "That concludes our ultimate guide to the Vice City docks."
            ),
            evidence=[
                EvidenceClip(
                    source_id="https://youtube.com/watch?v=src1",
                    start=600.0,
                    end=618.0,
                    role="evidence",
                    reason="Police boat chase around harbor pier",
                )
            ],
            use_in=["long"],
        ),
    ]

    shorts = [
        ShortPlan(
            kind="teaser",
            slot="teaser",
            point_indexes=[1],
            hook_title="Secret Harbor Vault Found",
            narration_intro=(
                "Bro check out what was hidden inside the cargo ship dock. "
                "AlphaGamer opened this secret vault door right on stream. "
                "Nobody knew this room even existed in the docks."
            ),
            callout="Watch the vault open!",
            narration_outro=(
                "Full breakdown on the channel! "
                "Which secret blew your mind the most bro? "
                "Tell me your favorite escape route in the comments."
            ),
        ),
        ShortPlan(
            kind="standalone",
            slot="short_a",
            point_indexes=[2],
            hook_title="Silent Warehouse Hack",
            narration_intro=(
                "Look at this guard patrol blind spot near the warehouse. "
                "DesiPlayer slipped past the warehouse basement door completely unseen. "
                "The security pattern has a massive flaw right here."
            ),
            callout="Watch the basement unlock!",
            narration_outro=(
                "Would you risk entering that warehouse bro? "
                "Have you ever pulled off a silent hack before? "
                "Tell me your craziest stealth moments in the comments below."
            ),
        ),
        ShortPlan(
            kind="standalone",
            slot="short_b",
            point_indexes=[3],
            hook_title="Insane Crane Bridge",
            narration_intro=(
                "Check out this crazy crane shortcut high above the docks. "
                "He hacked the crane control panel to move the container. "
                "This created an instant bridge across the harbor gap."
            ),
            callout="Look at the drop below!",
            narration_outro=(
                "Did you know about this crane shortcut bro? "
                "Would you dare make that container jump yourself? "
                "Let me know your thoughts in the comments."
            ),
        ),
    ]

    plan = TopicPlan(
        plan_id="plan_docks_test_01",
        topic_id="vice-city-docks-secrets",
        topic_title="Vice City Docks Secrets",
        game="GTA6",
        shape="numbered",
        language="english",
        status="draft",
        cold_open=(
            "You will not believe what players uncovered at the docks. "
            "Hidden vaults, secret tunnels, and insane escape routes were buried here. "
            "AlphaGamer and DesiPlayer broke down every single location. "
            "Nobody expected these cargo containers to hide so much loot. "
            "Today we reveal the entire secret network step by step. "
            "Get ready for five unbelievable dock discoveries."
        ),
        outro=(
            "That covers all five incredible secrets hidden in the docks. "
            "From underground tunnels to crane hacks, these routes change everything. "
            "Which of these escape routes was the craziest bro? "
            "Would you take the water tunnel or the speedboat getaway? "
            "Drop your thoughts and theories in the comments below. "
            "We will see you in the next investigation."
        ),
        points=points,
        shorts=shorts,
        video_title="Every Secret Hidden in the Vice City Docks",
        thumbnail_text="SECRET HARBOR VAULT",
        description="Full breakdown of hidden details and secret locations at the docks.",
    )
    normalize_plan_clips(cfg, plan)
    return plan


# ---------------------------------------------------------------------------
# 1. Models & Serialization Round-Trip
# ---------------------------------------------------------------------------

def test_T4_models_roundtrip(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)

    d = plan.to_dict()
    assert isinstance(d, dict)
    assert d["plan_id"] == "plan_docks_test_01"
    assert len(d["points"]) == 5
    assert len(d["shorts"]) == 3

    # Round trip
    reloaded = TopicPlan.from_dict(d)
    assert reloaded.plan_id == plan.plan_id
    assert reloaded.shape == plan.shape
    assert len(reloaded.points) == 5
    assert reloaded.points[0].heading == "#1 The Secret Vault"
    assert len(reloaded.points[0].evidence) == 1
    assert reloaded.points[0].evidence[0].source_id == plan.points[0].evidence[0].source_id
    assert len(reloaded.shorts) == 3
    assert reloaded.shorts[0].slot == "teaser"


# ---------------------------------------------------------------------------
# 2. Shape Rotation
# ---------------------------------------------------------------------------

def test_T4_shape_rotation(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, _ = sample_library
    ledger = TopicLedger(cfg)

    # Initial shape defaults to 'numbered'
    assert choose_shape(cfg, "GTA6") == "numbered"

    # Record numbered -> next must be countdown
    ledger.record("topic-1", "Topic One", "GTA6", shape="numbered")
    assert choose_shape(cfg, "GTA6") == "countdown"

    # Record countdown -> next must be story
    ledger.record("topic-2", "Topic Two", "GTA6", shape="countdown")
    assert choose_shape(cfg, "GTA6") == "story"

    # Record story -> next rotates back to numbered
    ledger.record("topic-3", "Topic Three", "GTA6", shape="story")
    assert choose_shape(cfg, "GTA6") == "numbered"


# ---------------------------------------------------------------------------
# 3. Plan Storage, Atomic Write, and Cache Index
# ---------------------------------------------------------------------------

def test_T4_plan_storage_atomic_and_index(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)

    file_path = save_plan(cfg, plan)
    assert file_path.exists()
    assert file_path.name == "plan.json"

    # Direct load by ID via index
    loaded = load_plan(cfg, plan.plan_id)
    assert loaded is not None
    assert loaded.plan_id == plan.plan_id
    assert loaded.topic_title == plan.topic_title

    # Delete index file; load_plan should scan <output_dir> and repopulate index
    cache_index = Path(cfg.get("general.cache_dir")) / "topic_plans.json"
    if cache_index.exists():
        cache_index.unlink()

    recovered = load_plan(cfg, plan.plan_id)
    assert recovered is not None
    assert recovered.plan_id == plan.plan_id
    assert cache_index.exists()

    # Unknown ID returns None
    assert load_plan(cfg, "non_existent_plan_id") is None


# ---------------------------------------------------------------------------
# 4. Planning Material Synthesis
# ---------------------------------------------------------------------------

def test_T4_planning_material(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, _ = sample_library
    material = planning_material(cfg, "GTA6", "Vice City docks cargo ship")

    assert material["game"] == "GTA6"
    assert "evidence_spans" in material
    assert len(material["evidence_spans"]) > 0
    span = material["evidence_spans"][0]
    assert "source_id" in span
    assert "start" in span
    assert "end" in span
    assert "duration" in span
    assert "transcript_text" in span
    assert material["shape"] in ("numbered", "countdown", "story")
    assert material["language"] in ("english", "hinglish")
    assert "narration_guide" in material
    assert "rules" in material
    assert "schema" in material


# ---------------------------------------------------------------------------
# 5. Validation: Fully Valid Plan Passes All Checks
# ---------------------------------------------------------------------------

def test_T4_validation_fully_valid_plan(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)

    problems = validate_plan(cfg, plan)
    assert problems == [], f"Expected 0 problems, got: {problems}"
    assert 480.0 <= plan.estimated_seconds <= 600.0
    assert plan.narration_seconds / plan.estimated_seconds >= 0.40


# ---------------------------------------------------------------------------
# 6. Validation: Every Contract Rule Has One Failing Plan
# ---------------------------------------------------------------------------

def test_T4_validation_fail_invalid_shape(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.shape = "invalid_shape_xyz"
    problems = validate_plan(cfg, plan)
    assert any("Invalid shape" in p for p in problems)


def test_T4_validation_fail_repeated_shape_from_ledger(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    ledger = TopicLedger(cfg)
    ledger.record("prev-topic", "Previous Topic", "GTA6", shape="numbered")

    plan = _build_valid_plan(cfg, index)
    plan.shape = "numbered"  # matches last ledger entry
    problems = validate_plan(cfg, plan)
    assert any("was used in the previous video" in p for p in problems)


def test_T4_validation_fail_points_count_numbered(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.points = plan.points[:3]  # Only 3 points (numbered requires 5-7)
    problems = validate_plan(cfg, plan)
    assert any("5–7 points" in p for p in problems)


def test_T4_validation_fail_points_count_story(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.shape = "story"
    for p in plan.points:
        p.heading = p.heading.replace("#", "").strip()
    plan.points = plan.points[:2]  # Only 2 points (story requires 4-6)
    problems = validate_plan(cfg, plan)
    assert any("4–6 points" in p for p in problems)


def test_T4_validation_fail_headings_missing_hash_numbered(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.points[0].heading = "Missing Hash Prefix"
    problems = validate_plan(cfg, plan)
    assert any("must contain a '#' chapter prefix" in p for p in problems)


def test_T4_validation_fail_headings_has_hash_story(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.shape = "story"
    # Keep "#1" in heading
    problems = validate_plan(cfg, plan)
    assert any("Story shape heading" in p and "must not contain '#'" in p for p in problems)


def test_T4_validation_fail_countdown_not_descending(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.shape = "countdown"
    # Points in ascending order (1, 2, 3...) instead of descending (5, 4, 3...)
    problems = validate_plan(cfg, plan)
    assert any("Countdown shape points must be ordered high to low" in p for p in problems)


def test_T4_validation_fail_empty_cold_open_or_outro(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.cold_open = ""
    problems = validate_plan(cfg, plan)
    assert any("cold_open cannot be empty" in p for p in problems)


def test_T4_validation_fail_clip_source_not_in_library(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.points[0].evidence[0].source_id = "https://youtube.com/watch?v=unknown_src_xyz"
    problems = validate_plan(cfg, plan)
    assert any("not found in library" in p for p in problems)


def test_T4_validation_fail_clip_duration_exceeds_20s(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.points[0].evidence[0].start = 10.0
    plan.points[0].evidence[0].end = 35.0  # 25 seconds > 20s
    problems = validate_plan(cfg, plan)
    assert any("exceeds max clip duration of 20.0s" in p for p in problems)


def test_T4_validation_official_10s_cap(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    # Use official source clip of 12 seconds (> 10s cap)
    plan.points[0].evidence = [
        EvidenceClip(
            source_id="https://youtube.com/watch?v=src_official",
            start=0.0,
            end=12.0,
            kind="official",
            role="evidence",
            reason="Official trailer footage",
        )
    ]
    problems = validate_plan(cfg, plan)
    assert any("exceeds max official clip duration of 10.0s" in p for p in problems)


def test_T4_validation_fail_duration_bounds(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)

    # Make narrations extremely short to trigger < 480s duration failure
    for p in plan.points:
        p.narration = "Very short line here."
    problems = validate_plan(cfg, plan)
    assert any("below minimum 480.0s" in p for p in problems)


def test_T4_validation_fail_narration_ratio_low(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    # Set config min_narration_ratio to 0.99 so even long narration fails
    cfg._data["topics"]["min_narration_ratio"] = 0.99
    problems = validate_plan(cfg, plan)
    assert any("Narration share" in p and "below minimum" in p for p in problems)


def test_T4_validation_fail_shorts_count_and_slots(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.shorts = plan.shorts[:2]  # Only 2 shorts instead of 3
    problems = validate_plan(cfg, plan)
    assert any("exactly 3 Shorts" in p for p in problems)


def test_T4_validation_fail_teaser_missing_pointer(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.shorts[0].narration_outro = "Thanks for watching guys see you later!"
    problems = validate_plan(cfg, plan)
    assert any("Teaser outro must end with a pointer to the full video" in p for p in problems)


def test_T4_validation_fail_standalone_short_references_long_video(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.shorts[1].narration_outro = "Watch the full breakdown on the channel!"
    problems = validate_plan(cfg, plan)
    assert any("Standalone Short 'short_a' must not reference the long video" in p for p in problems)


def test_T4_validation_short_overlap_detection(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)

    # Point 2 and Point 3 both use source 2 with overlapping intervals (180-195 and 185-195)
    plan.points[1].evidence = [
        EvidenceClip(source_id="https://youtube.com/watch?v=src2", start=180.0, end=195.0, role="evidence")
    ]
    plan.points[2].evidence = [
        EvidenceClip(source_id="https://youtube.com/watch?v=src2", start=185.0, end=195.0, role="evidence")
    ]
    # short_a uses point 2, short_b uses point 3
    plan.shorts[1].point_indexes = [2]
    plan.shorts[2].point_indexes = [3]

    problems = validate_plan(cfg, plan)
    assert any("share overlapping footage" in p for p in problems)


def test_T4_validation_fail_short_duration_out_of_bounds(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    # Short too short (< 35s)
    plan.shorts[0].narration_intro = "Hi."
    plan.shorts[0].callout = "Look."
    plan.shorts[0].narration_outro = "Full breakdown on the channel."
    plan.points[0].evidence = [
        EvidenceClip(source_id="https://youtube.com/watch?v=src1", start=10.0, end=15.0, role="evidence")
    ]
    problems = validate_plan(cfg, plan)
    assert any("Short 'teaser' estimated duration" in p and "below minimum" in p for p in problems)


def test_T4_validation_fail_banned_generic_phrases(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.points[0].narration = "Wait for it guys this is an absolutely insane crazy moment."
    problems = validate_plan(cfg, plan)
    assert any("uses banned phrase" in p or "banned generic phrase" in p for p in problems)


# ---------------------------------------------------------------------------
# 7. Estimate Math Verification
# ---------------------------------------------------------------------------

def test_T4_estimate_math(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)

    validate_plan(cfg, plan)

    cold_words = len(plan.cold_open.split())
    outro_words = len(plan.outro.split())
    pts_words = sum(len(p.narration.split()) for p in plan.points)
    expected_narr_sec = round((cold_words + pts_words + outro_words) / 2.7, 1)

    expected_cards_sec = 2.5 * len(plan.points)
    expected_clips_sec = sum(c.end - c.start for p in plan.points for c in p.evidence)
    expected_total_sec = round(expected_narr_sec + expected_cards_sec + expected_clips_sec, 1)

    assert plan.narration_seconds == expected_narr_sec
    assert plan.estimated_seconds == expected_total_sec


# ---------------------------------------------------------------------------
# 8. Local Planner Retry & Keep Best
# ---------------------------------------------------------------------------

def test_T4_local_planner_retry_and_keep_best(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    valid_plan = _build_valid_plan(cfg, index)

    # Attempt 1: 3 problems
    plan_attempt_1 = valid_plan.to_dict()
    plan_attempt_1["shape"] = "invalid_shape_1"
    plan_attempt_1["cold_open"] = ""

    # Attempt 2: 1 problem (best attempt)
    plan_attempt_2 = valid_plan.to_dict()
    plan_attempt_2["shape"] = "invalid_shape_2"

    # Attempt 3: 2 problems
    plan_attempt_3 = valid_plan.to_dict()
    plan_attempt_3["shape"] = "invalid_shape_3"
    plan_attempt_3["outro"] = ""

    mock_client = MagicMock()
    mock_client.chat.side_effect = [
        {"message": {"content": json.dumps(plan_attempt_1)}},
        {"message": {"content": json.dumps(plan_attempt_2)}},
        {"message": {"content": json.dumps(plan_attempt_3)}},
    ]

    result = plan_topic_local(cfg, "GTA6", "Vice City Docks Secrets", client=mock_client)

    assert mock_client.chat.call_count == 3
    assert result.status == "draft"
    # Result should be attempt 2 because it had the fewest problems
    assert result.shape == "invalid_shape_2"
    assert len(result.problems) == 1


# ---------------------------------------------------------------------------
# 9. MCP Tool Workflow (Start -> Submit -> Update -> Approve)
# ---------------------------------------------------------------------------

def test_T4_mcp_flow_start_submit_update_approve(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    valid_plan = _build_valid_plan(cfg, index)

    with patch("streamclipper.mcp_server.server.Config", return_value=cfg):
        # Step 1: start_topic_plan
        start_res_str = start_topic_plan("GTA6", "Docks Mystery")
        start_res = json.loads(start_res_str)
        assert "plan_id" in start_res
        assert "material" in start_res
        fresh_plan_id = start_res["plan_id"]

        # Step 2: submit_topic_plan with an intentional problem (teaser missing pointer)
        plan_dict = valid_plan.to_dict()
        plan_dict["plan_id"] = fresh_plan_id
        plan_dict["shorts"][0]["narration_outro"] = "Thank you guys for watching!"

        submit_res_str = submit_topic_plan(json.dumps(plan_dict))
        submit_res = json.loads(submit_res_str)
        assert submit_res["valid"] is False
        assert len(submit_res["problems"]) > 0
        assert submit_res["status"] == "draft"

        # Step 3: approve_topic_plan refused while invalid
        approve_fail_str = approve_topic_plan(fresh_plan_id)
        approve_fail = json.loads(approve_fail_str)
        assert "error" in approve_fail
        assert "Cannot approve topic plan with validation problems" in approve_fail["error"]

        # Step 4: update_topic_plan fixes the problem
        patch_payload = {
            "shorts": [
                {
                    "kind": "teaser",
                    "slot": "teaser",
                    "point_indexes": [1],
                    "hook_title": "Secret Harbor Vault Found",
                    "narration_intro": (
                        "Bro check out what was hidden inside the cargo ship dock. "
                        "AlphaGamer opened this secret vault door right on stream. "
                        "Nobody knew this room even existed in the docks."
                    ),
                    "callout": "Watch the vault open!",
                    "narration_outro": (
                        "Full breakdown on the channel! "
                        "Which secret blew your mind the most bro? "
                        "Tell me your favorite escape route in the comments."
                    ),
                },
                plan_dict["shorts"][1],
                plan_dict["shorts"][2],
            ]
        }
        update_res_str = update_topic_plan(fresh_plan_id, json.dumps(patch_payload))
        update_res = json.loads(update_res_str)
        assert update_res["valid"] is True, f"Expected update to be valid, got problems: {update_res.get('problems')}"
        assert len(update_res["problems"]) == 0

        # Step 5: get_topic_plan retrieves valid plan
        get_res_str = get_topic_plan(fresh_plan_id)
        get_res = json.loads(get_res_str)
        assert get_res["valid"] is True
        assert get_res["plan_id"] == fresh_plan_id

        # Step 6: approve_topic_plan succeeds
        approve_ok_str = approve_topic_plan(fresh_plan_id)
        approve_ok = json.loads(approve_ok_str)
        assert approve_ok["status"] == "approved"
        assert "approved_at" in approve_ok

        # Step 7: update_topic_plan refused after approval
        update_after_res_str = update_topic_plan(fresh_plan_id, json.dumps({"cold_open": "New open"}))
        update_after_res = json.loads(update_after_res_str)
        assert "error" in update_after_res
        assert "already approved" in update_after_res["error"]

        # Step 8: Ledger records approved topic
        ledger = TopicLedger(cfg)
        ledger_entries = ledger.entries("GTA6")
        assert len(ledger_entries) == 1
        assert ledger_entries[0]["plan_id"] == fresh_plan_id
        assert ledger_entries[0]["shape"] == valid_plan.shape


# ---------------------------------------------------------------------------
# 10. CLI Subcommands (plan, show, validate, approve)
# ---------------------------------------------------------------------------

def test_T4_cli_subcommands(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    valid_plan = _build_valid_plan(cfg, index)
    save_plan(cfg, valid_plan)

    cfg_file = Path(cfg.get("general.cache_dir")).parent / "config.yaml"
    runner = CliRunner()

    # 1. show command
    res_show = runner.invoke(cli, ["--config", str(cfg_file), "topics", "show", valid_plan.plan_id])
    assert res_show.exit_code == 0, f"res_show failed: {res_show.output}"
    assert valid_plan.plan_id in res_show.output
    assert "#1 The Secret Vault" in res_show.output

    # 2. validate command (valid)
    res_val = runner.invoke(cli, ["--config", str(cfg_file), "topics", "validate", valid_plan.plan_id])
    assert res_val.exit_code == 0, f"res_val failed: {res_val.output}"
    assert "is valid" in res_val.output

    # 3. approve command
    res_app = runner.invoke(cli, ["--config", str(cfg_file), "topics", "approve", valid_plan.plan_id])
    assert res_app.exit_code == 0, f"res_app failed: {res_app.output}"
    assert "approved successfully" in res_app.output

    # 4. validate non-existent ID fails
    res_missing = runner.invoke(cli, ["--config", str(cfg_file), "topics", "validate", "missing_plan_id"])
    assert res_missing.exit_code != 0

    # 5. plan command supports both positional and option arguments
    with patch("streamclipper.topics.planner.plan_topic_local", return_value=valid_plan) as mock_plan:
        res_plan_pos = runner.invoke(cli, ["--config", str(cfg_file), "topics", "plan", "GTA6", "Secret Vault"])
        assert res_plan_pos.exit_code == 0, f"res_plan_pos failed: {res_plan_pos.output}"
        assert mock_plan.called

        mock_plan.reset_mock()
        res_plan_opt = runner.invoke(cli, ["--config", str(cfg_file), "topics", "plan", "--game", "GTA6", "--topic", "Secret Vault"])
        assert res_plan_opt.exit_code == 0, f"res_plan_opt failed: {res_plan_opt.output}"
        assert mock_plan.called


# ---------------------------------------------------------------------------
# 11. Additional Edge Cases & Contract Safeguards
# ---------------------------------------------------------------------------

def test_T4_validation_fail_point_narration_sentence_too_long(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.points[0].narration = (
        "This is an excessively long and verbose sentence that contains way more than the maximum allowable limit of fourteen words per sentence. "
        "Short sentence here."
    )
    problems = validate_plan(cfg, plan)
    assert any("has a sentence of" in p and "max 14" in p for p in problems)


def test_T4_validation_fail_point_narration_ungrounded(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.points[0].narration = "Purple aliens hovered silently above the downtown skyline last night."
    problems = validate_plan(cfg, plan)
    assert any("NOT GROUNDED" in p for p in problems)


def test_T4_validation_fail_standalone_short_references_in_hook_title(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    plan = _build_valid_plan(cfg, index)
    plan.shorts[1].hook_title = "Full breakdown on the harbor dock"
    problems = validate_plan(cfg, plan)
    assert any("must not reference the long video" in p for p in problems)


def test_T4_planning_material_language_override(sample_library: tuple[Config, LibraryIndex]) -> None:
    cfg, index = sample_library
    # Explicit language override argument
    mat1 = planning_material(cfg, "GTA6", "Docks Mystery", language="hinglish")
    assert mat1["language"] == "hinglish"

    # Config-driven transformation narration language override
    cfg._data.setdefault("transformation", {})["narration_language"] = "hinglish"
    mat2 = planning_material(cfg, "GTA6", "Docks Mystery")
    assert mat2["language"] == "hinglish"
    cfg._data["transformation"]["narration_language"] = "auto"


def test_T4_models_zero_values_preserved() -> None:
    # Point index=0 preserved
    p = TopicPoint.from_dict({"index": 0, "heading": "Prologue", "narration": "Intro text"})
    assert p.index == 0

    # Evidence clip start=0.0 and end=0.0 preserved
    c = EvidenceClip.from_dict({"source_id": "src1", "start": 0.0, "end": 10.0})
    assert c.start == 0.0
    assert c.end == 10.0

    # TopicPlan estimated_seconds=0.0 and narration_seconds=0.0 preserved
    plan = TopicPlan.from_dict({"plan_id": "p0", "game": "GTA6", "estimated_seconds": 0.0, "narration_seconds": 0.0})
    assert plan.estimated_seconds == 0.0
    assert plan.narration_seconds == 0.0

