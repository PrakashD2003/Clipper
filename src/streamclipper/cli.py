"""CLI entry point for StreamClipper."""
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import Config

console = Console()


def _configure_logging(config: Config, verbose: bool) -> None:
    """Configure root and package logging from --verbose or general.log_level."""
    if verbose:
        level = logging.DEBUG
    else:
        level_name = str(config.get("general.log_level", "INFO")).upper()
        level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s [%(name)s] %(message)s")
    logging.getLogger("streamclipper").setLevel(level)


@click.group()
@click.version_option(version=__version__, prog_name="streamclipper")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to config.yaml",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx: click.Context, config: str | None, verbose: bool) -> None:
    """🎮 StreamClipper — AI-powered stream clipping tool"""
    ctx.ensure_object(dict)
    cfg = Config(Path(config) if config else None)
    _configure_logging(cfg, verbose)
    ctx.obj["config"] = cfg
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("url")
@click.option("--max-clips", "-n", default=None, type=int, help="Max clips to generate")
@click.option("--game", "-g", default=None, help="Game name (overrides config)")
@click.option("--streamer", "-s", default=None, help="Streamer name (auto-detected if omitted)")
@click.option("--interactive/--no-interactive", default=True, help="Prompt to select highlights")
@click.option(
    "--mode",
    type=click.Choice(["auto", "narrative", "story", "breakdown", "raw"], case_sensitive=False),
    default=None,
    help="YPP transformation mode (overrides config)",
)
@click.option(
    "--voice-intro",
    "voice_intro",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to custom human intro voiceover WAV",
)
@click.option(
    "--voice-outro",
    "voice_outro",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to custom human outro voiceover WAV",
)
@click.option(
    "--voice-breakdown",
    "voice_breakdown",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to custom human breakdown voiceover WAV",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Ignore cached pipeline state and re-run from scratch",
)
@click.pass_context
def clip(
    ctx: click.Context,
    url: str,
    max_clips: int | None,
    game: str | None,
    streamer: str | None,
    interactive: bool,
    mode: str | None,
    voice_intro: Path | None,
    voice_outro: Path | None,
    force: bool,
    voice_breakdown: Path | None = None,
) -> None:
    """Clip highlights from a stream URL.

    Example: streamclipper clip "https://youtube.com/watch?v=XXXXX" --mode auto
    """
    from .pipeline import Pipeline

    config: Config = ctx.obj["config"]
    normalized_mode = mode.lower() if mode else None
    if voice_intro is not None and not voice_intro.exists():
        console.print(f"[yellow]⚠ Intro voiceover file not found yet:[/] {voice_intro}")
    if voice_outro is not None and not voice_outro.exists():
        console.print(f"[yellow]⚠ Outro voiceover file not found yet:[/] {voice_outro}")
    if voice_breakdown is not None and not voice_breakdown.exists():
        console.print(f"[yellow]⚠ Breakdown voiceover file not found yet:[/] {voice_breakdown}")

    console.print(f"[bold green]🎬 Processing stream:[/] {url}")

    pipeline = Pipeline(config)
    call_kwargs: dict[str, Any] = {
        "url": url,
        "max_clips": max_clips,
        "game": game,
        "streamer": streamer,
        "interactive": interactive,
        "mode": normalized_mode,
        "voice_intro_path": voice_intro,
        "voice_outro_path": voice_outro,
    }
    if voice_breakdown is not None:
        call_kwargs["voice_breakdown_path"] = voice_breakdown
    if force:
        call_kwargs["force"] = True

    try:
        clips = asyncio.run(pipeline.process_stream(**call_kwargs))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if not clips:
        console.print("[yellow]No clips were generated for this stream.[/]")
        return

    console.print(f"[bold green]✅ Generated {len(clips)} clip(s):[/]")
    for idx, result in enumerate(clips, 1):
        music_flag = " [red]⚠️ MUSIC_DETECTED[/]" if result.metadata.music_risk == "HIGH_MUSIC_DETECTED" else ""
        console.print(
            f"  {idx}. [cyan]{result.video_path}[/]{music_flag} "
            f"([dim]{result.metadata.transformation_mode}, {result.duration_seconds:.1f}s[/])"
        )


@cli.command()
@click.option("--game", "-g", default=None, help="Game to search for")
@click.option("--max-streams", "-n", default=None, type=int, help="Number of streams")
@click.option("--max-clips", default=None, type=int, help="Clips per stream")
@click.option(
    "--mode",
    type=click.Choice(["auto", "narrative", "story", "breakdown", "raw"], case_sensitive=False),
    default=None,
    help="YPP transformation mode (overrides config)",
)
@click.option(
    "--voice-intro",
    "voice_intro",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to custom human intro voiceover WAV",
)
@click.option(
    "--voice-outro",
    "voice_outro",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to custom human outro voiceover WAV",
)
@click.option(
    "--voice-breakdown",
    "voice_breakdown",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to custom human breakdown voiceover WAV",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Ignore cached pipeline state and re-run from scratch",
)
@click.pass_context
def auto(
    ctx: click.Context,
    game: str | None,
    max_streams: int | None,
    max_clips: int | None,
    mode: str | None,
    voice_intro: Path | None,
    voice_outro: Path | None,
    force: bool,
    voice_breakdown: Path | None = None,
) -> None:
    """Auto-discover top streams and clip them all.

    Example: streamclipper auto --game "GTA 6" --max-streams 5 --mode auto
    """
    from .discovery.youtube import YouTubeDiscovery
    from .pipeline import Pipeline

    config: Config = ctx.obj["config"]
    target_game = game or str(config.get("discovery.default_game", "GTA 6"))
    normalized_mode = mode.lower() if mode else None
    if voice_intro is not None and not voice_intro.exists():
        console.print(f"[yellow]⚠ Intro voiceover file not found yet:[/] {voice_intro}")
    if voice_outro is not None and not voice_outro.exists():
        console.print(f"[yellow]⚠ Outro voiceover file not found yet:[/] {voice_outro}")
    if voice_breakdown is not None and not voice_breakdown.exists():
        console.print(f"[yellow]⚠ Breakdown voiceover file not found yet:[/] {voice_breakdown}")

    console.print(f"[bold green]🔍 Discovering top {target_game} streams...[/]")

    discovery = YouTubeDiscovery(config)
    if max_streams is not None:
        discovery.max_streams = max(0, max_streams)
    streams = discovery.discover(game=target_game)
    if max_streams is not None:
        streams = streams[: max(0, max_streams)]

    if not streams:
        console.print(f"[yellow]No matching {target_game} streams found.[/]")
        return

    console.print(
        f"[bold green]🚀 Processing {len(streams)} stream(s) in parallel...[/]"
    )
    max_concurrent = int(config.get("general.max_concurrent_streams", 2))
    pipeline = Pipeline(config)
    parallel_kwargs: dict[str, Any] = {
        "max_concurrent": max_concurrent,
        "max_clips": max_clips,
        "game": target_game,
        "mode": normalized_mode,
        "voice_intro_path": voice_intro,
        "voice_outro_path": voice_outro,
    }
    if voice_breakdown is not None:
        parallel_kwargs["voice_breakdown_path"] = voice_breakdown
    if force:
        parallel_kwargs["force"] = True
    results = asyncio.run(
        pipeline.process_streams_parallel(
            [s.url for s in streams],
            **parallel_kwargs,
        )
    )

    total_clips = 0
    failures = 0
    for stream, outcome in zip(streams, results):
        if isinstance(outcome, BaseException):
            failures += 1
            console.print(f"  [red]✗[/] {stream.title or stream.url}: {outcome}")
        else:
            total_clips += len(outcome)
            console.print(
                f"  [green]✓[/] {stream.title or stream.url}: {len(outcome)} clip(s)"
            )

    console.print(
        f"[bold green]✅ Finished auto-run:[/] {total_clips} clip(s) generated across "
        f"{len(streams) - failures}/{len(streams)} stream(s)."
    )
    if failures:
        ctx.exit(1)


@cli.command()
@click.pass_context
def setup(ctx: click.Context) -> None:
    """Check dependencies, detect hardware profile, and report system readiness.

    Example: streamclipper setup
    """
    config: Config = ctx.obj["config"]
    console.print("[bold]🔧 Checking system dependencies...[/]")
    _check_dependency("yt-dlp", "yt_dlp")
    _check_dependency("FFmpeg", None, cmd="ffmpeg -version")
    _check_dependency("WhisperX", "whisperx")
    _check_dependency("SoundFile", "soundfile")
    _check_dependency("OpenCV", "cv2")
    _check_dependency("MoviePy v2", "moviepy")
    _check_dependency("Ollama SDK", "ollama")
    _check_dependency("MCP SDK v2 (MCPServer)", "mcp.server.mcpserver")
    _check_gpu()
    _render_hardware_profile(config)


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show status of cached/in-progress pipelines.

    Example: streamclipper status
    """
    from .cache import StepCache

    cache = StepCache(ctx.obj["config"])
    pipelines = cache.list_all()

    if not pipelines:
        console.print("[dim]No cached pipelines found.[/]")
        return

    table = Table(title="Pipeline Status")
    table.add_column("Stream", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Highlights", style="yellow")
    table.add_column("Clips", style="magenta")

    for p in pipelines:
        table.add_row(
            p.stream_info.title[:40] or p.stream_info.url[:40],
            p.status.name.lower(),
            str(len(p.highlights)),
            str(len(p.clips)),
        )
    console.print(table)


def _check_dependency(name: str, module: str | None, cmd: str | None = None) -> None:
    """Check if a dependency is installed."""
    try:
        if module:
            __import__(module)
        if cmd:
            subprocess.run(cmd.split(), capture_output=True, check=True, timeout=10)
        console.print(f"  [green]✓[/] {name}")
    except (
        ImportError,
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
    ):
        console.print(f"  [red]✗[/] {name} — not installed")


def _check_gpu() -> None:
    """Check CUDA GPU availability."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / 1e9
            cc = f"{props.major}.{props.minor}"
            console.print(
                f"  [green]✓[/] GPU: {props.name} ({vram:.1f}GB VRAM, CC {cc})"
            )
        else:
            console.print("  [yellow]⚠[/] No CUDA GPU detected — will use CPU (slower)")
    except ImportError:
        console.print("  [yellow]⚠[/] PyTorch not installed — GPU check skipped")


def _render_hardware_profile(config: Config) -> None:
    """Display the resolved HardwareProfile and active config values."""
    hw = config.hardware_profile
    table = Table(title=f"🖥️ Resolved Hardware Profile ({hw.tier})")
    table.add_column("Setting", style="cyan")
    table.add_column("Active Value", style="green")
    table.add_column("Source", style="dim")

    rows = [
        (
            "GPU / VRAM / CC",
            f"{hw.gpu_name} ({hw.vram_gb:.1f} GB, CC {hw.compute_capability})",
            "detected",
        ),
        (
            "Tensor Cores / NVENC",
            f"Tensor={hw.has_tensor_cores}, NVENC={hw.has_nvenc}",
            "detected",
        ),
        (
            "transcription.device",
            str(config.get("transcription.device")),
            "auto" if config.is_auto_resolved("transcription.device") else "user override",
        ),
        (
            "transcription.model_size",
            str(config.get("transcription.model_size")),
            "auto" if config.is_auto_resolved("transcription.model_size") else "user override",
        ),
        (
            "transcription.compute_type",
            str(config.get("transcription.compute_type")),
            "auto" if config.is_auto_resolved("transcription.compute_type") else "user override",
        ),
        (
            "transcription.batch_size",
            str(config.get("transcription.batch_size")),
            "auto" if config.is_auto_resolved("transcription.batch_size") else "user override",
        ),
        (
            "llm.model",
            str(config.get("llm.model")),
            "auto" if config.is_auto_resolved("llm.model") else "user override",
        ),
        (
            "video.codec",
            str(config.get("video.codec")),
            "auto" if config.is_auto_resolved("video.codec") else "user override",
        ),
    ]
    for setting, value, source in rows:
        table.add_row(setting, value, source)
    console.print(table)


@cli.group(name="library")
def library_group() -> None:
    """Manage local source library and search index for topic videos."""
    pass


@library_group.command(name="add")
@click.argument("url")
@click.option("--game", "-g", required=True, help="Game name")
@click.option(
    "--kind",
    "-k",
    type=click.Choice(["stream", "gameplay", "official"], case_sensitive=False),
    default="stream",
    help="Source kind",
)
@click.option("--channel", "-s", default=None, help="Channel / streamer name")
@click.option("--force", "-f", is_flag=True, default=False, help="Force re-download / re-index")
@click.pass_context
def library_add(
    ctx: click.Context,
    url: str,
    game: str,
    kind: str,
    channel: str | None,
    force: bool,
) -> None:
    """Add and index a source video."""
    from streamclipper.library import add_source

    cfg: Config = ctx.obj["config"]
    try:
        source = asyncio.run(
            add_source(cfg, url, game=game, kind=kind, channel=channel, force=force)
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    console.print(
        f"[green]Indexed source {source.source_id}: {source.title or source.url} ({source.segment_count} segments)[/green]"
    )


@library_group.command(name="import-cache")
@click.option("--game", "-g", default=None, help="Filter cached states by game")
@click.pass_context
def library_import_cache(ctx: click.Context, game: str | None) -> None:
    """Index already analyzed cached streams into the library."""
    from streamclipper.library import index_cached_states

    cfg: Config = ctx.obj["config"]
    sources = index_cached_states(cfg, game=game)
    console.print(f"[green]Imported {len(sources)} sources from cache into library.[/green]")


@library_group.command(name="list")
@click.option("--game", "-g", default=None, help="Filter sources by game")
@click.pass_context
def library_list(ctx: click.Context, game: str | None) -> None:
    """List indexed source videos."""
    from streamclipper.library import LibraryIndex

    cfg: Config = ctx.obj["config"]
    index = LibraryIndex(cfg)
    sources = index.list_sources(game=game)
    if not sources:
        console.print("No sources found.")
        return

    table = Table(title="Library Sources")
    table.add_column("Channel")
    table.add_column("Kind")
    table.add_column("Title")
    table.add_column("Duration")
    table.add_column("Segments", justify="right")
    table.add_column("Indexed")
    for s in sources:
        if s.duration_seconds >= 3600:
            dur_h, rem = divmod(s.duration_seconds, 3600)
            dur_m, dur_s = divmod(rem, 60)
            dur_str = f"{dur_h:d}:{dur_m:02d}:{dur_s:02d}"
        else:
            dur_m, dur_s = divmod(s.duration_seconds, 60)
            dur_str = f"{dur_m:02d}:{dur_s:02d}"
        table.add_row(
            s.channel,
            s.kind,
            s.title or s.url,
            dur_str,
            str(s.segment_count),
            s.indexed_at[:19].replace("T", " ") if s.indexed_at else "",
        )
    console.print(table)


@library_group.command(name="search")
@click.argument("query")
@click.option("--game", "-g", default=None, help="Filter by game")
@click.option(
    "--kind",
    "-k",
    "kinds",
    multiple=True,
    default=None,
    help="Filter by kind (repeatable)",
)
@click.option(
    "--channel",
    "-s",
    "channels",
    multiple=True,
    default=None,
    help="Filter by channel (repeatable)",
)
@click.option("--limit", "-l", default=20, type=int, help="Max results to return")
@click.pass_context
def library_search(
    ctx: click.Context,
    query: str,
    game: str | None,
    kinds: tuple[str, ...],
    channels: tuple[str, ...],
    limit: int,
) -> None:
    """Search indexed library for speech, chat, and signal moments."""
    from streamclipper.library import LibraryIndex

    cfg: Config = ctx.obj["config"]
    index = LibraryIndex(cfg)
    hits = index.search(
        query,
        game=game,
        kinds=list(kinds) if kinds else None,
        channels=list(channels) if channels else None,
        limit=limit,
    )
    if not hits:
        console.print("No matching segments found.")
        return

    table = Table(title=f"Search Results: {query}")
    table.add_column("Score", justify="right")
    table.add_column("Channel")
    table.add_column("Kind")
    table.add_column("Time")
    table.add_column("Snippet")
    table.add_column("Heat", justify="right")
    for hit in hits:
        start_m, start_s = divmod(int(hit.start), 60)
        end_m, end_s = divmod(int(hit.end), 60)
        time_span = f"{start_m:02d}:{start_s:02d}–{end_m:02d}:{end_s:02d}"
        table.add_row(
            f"{hit.score:.2f}",
            hit.channel,
            hit.kind,
            time_span,
            hit.snippet,
            f"{hit.heat:.2f}",
        )
    console.print(table)


@library_group.command(name="remove")
@click.argument("source_id")
@click.pass_context
def library_remove(ctx: click.Context, source_id: str) -> None:
    """Remove a source video and its indexed segments from the library."""
    from streamclipper.library import LibraryIndex

    cfg: Config = ctx.obj["config"]
    index = LibraryIndex(cfg)
    removed = index.remove_source(source_id)
    if removed:
        console.print(f"[green]Removed source {source_id} from library.[/green]")
    else:
        console.print(f"[yellow]Source {source_id} not found.[/yellow]")


@library_group.command(name="frames")
@click.argument("source_id")
@click.pass_context
def library_frames(ctx: click.Context, source_id: str) -> None:
    """Download low-res proxy and build frame contact sheets."""
    from streamclipper.library.frames import build_contact_sheets, download_proxy_video
    from streamclipper.library.index import LibraryIndex

    cfg: Config = ctx.obj["config"]
    index = LibraryIndex(cfg)
    source = index.get_source(source_id)
    if not source:
        raise click.ClickException(f"Source '{source_id}' not found in library.")

    temp_dir = Path(cfg.get("general.temp_dir", "./temp"))
    frames_dir = temp_dir / "frames" / source_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    proxy_height = int(cfg.get("library.frames.proxy_height", 360))
    console.print(f"[cyan]Downloading {proxy_height}p proxy video for {source.title or source_id}...[/cyan]")
    download_proxy_video(source.url, frames_dir, height=proxy_height)

    console.print("[cyan]Generating contact sheets...[/cyan]")
    sheets = build_contact_sheets(cfg, source_id)
    console.print(f"[green]Built {len(sheets)} contact sheets in {frames_dir}[/green]")


@library_group.command(name="describe")
@click.argument("source_id")
@click.option("--limit", "-l", default=None, type=int, help="Limit number of sheets to describe")
@click.pass_context
def library_describe(ctx: click.Context, source_id: str, limit: int | None) -> None:
    """Run local vision model over contact sheets and store notes."""
    from streamclipper.library.frames import build_contact_sheets
    from streamclipper.library.index import LibraryIndex
    from streamclipper.library.vision import VisionUnavailable, describe_sheet_ollama

    cfg: Config = ctx.obj["config"]
    index = LibraryIndex(cfg)
    source = index.get_source(source_id)
    if not source:
        raise click.ClickException(f"Source '{source_id}' not found in library.")

    sheets = build_contact_sheets(cfg, source_id)
    if not sheets:
        console.print(f"[yellow]No contact sheets found for {source_id}. Run 'library frames {source_id}' first.[/yellow]")
        return

    if limit is not None and limit > 0:
        sheets = sheets[:limit]

    vision_model = str(cfg.get("library.vision.model", "qwen2.5vl:3b") or "")
    console.print(f"[cyan]Describing {len(sheets)} sheets using vision model '{vision_model}'...[/cyan]")

    total_notes = 0
    try:
        for idx, sheet in enumerate(sheets, start=1):
            console.print(f"Describing sheet {idx}/{len(sheets)} ({sheet.start:.0f}s - {sheet.end:.0f}s)...")
            notes = describe_sheet_ollama(cfg, sheet)
            if notes:
                added = index.add_frame_notes(source_id, notes)
                total_notes += added
    except VisionUnavailable as exc:
        console.print(f"[red]{exc}[/red]")
        if vision_model:
            console.print(f"[yellow]To enable local vision, install Ollama and run: ollama pull {vision_model}[/yellow]")
        else:
            console.print("[yellow]To enable local vision, set library.vision.model in config.yaml and run: ollama pull <model>[/yellow]")
        return

    console.print(f"[green]Successfully stored {total_notes} frame notes for {source_id}.[/green]")


@cli.group(name="topics")
def topics_group() -> None:
    """Discover, evaluate, and manage topic video ideas."""
    pass


@topics_group.command(name="suggest")
@click.argument("game_arg", required=False, default=None, metavar="[GAME]")
@click.option("--game", "-g", default=None, help="Game name")
@click.option("--limit", "-l", default=10, type=int, help="Maximum topics to suggest")
@click.pass_context
def topics_suggest(ctx: click.Context, game_arg: str | None, game: str | None, limit: int) -> None:
    """Suggest topic video ideas with evidence counts and source diversity."""
    from streamclipper.topics.discover import suggest_topics_local

    resolved_game = (game or game_arg or "").strip()
    if not resolved_game:
        raise click.UsageError("Missing required game name. Provide either [GAME] argument or --game/-g option.")

    cfg: Config = ctx.obj["config"]
    candidates = suggest_topics_local(cfg, resolved_game, limit=limit)
    if not candidates:
        console.print(f"[yellow]No viable topic candidates found for '{resolved_game}'.[/yellow]")
        return

    table = Table(title=f"Suggested Topic Videos — {resolved_game}")
    table.add_column("Title", style="cyan")
    table.add_column("Evidence", justify="right")
    table.add_column("Sources", justify="right")
    table.add_column("Channels", justify="right")
    table.add_column("Already Made", justify="center")

    for c in candidates:
        made_str = "[yellow]Yes[/yellow]" if c.already_made else "[green]No[/green]"
        table.add_row(
            c.title,
            str(len(c.evidence)),
            str(c.source_count),
            str(c.channel_count),
            made_str,
        )
    console.print(table)


@topics_group.command(name="ledger")
@click.argument("game_arg", required=False, default=None, metavar="[GAME]")
@click.option("--game", "-g", default=None, help="Filter ledger entries by game")
@click.pass_context
def topics_ledger(ctx: click.Context, game_arg: str | None, game: str | None) -> None:
    """View produced topic videos recorded in the ledger."""
    from streamclipper.topics.ledger import TopicLedger

    resolved_game = (game or game_arg or "").strip()
    cfg: Config = ctx.obj["config"]
    ledger = TopicLedger(cfg)
    entries = ledger.entries(resolved_game)
    if not entries:
        console.print("No ledger entries found.")
        return

    title_str = f"Topic Ledger — {resolved_game}" if resolved_game else "Topic Ledger"
    table = Table(title=title_str)
    table.add_column("Topic ID", style="dim")
    table.add_column("Title", style="cyan")
    table.add_column("Game", style="green")
    table.add_column("Shape", style="magenta")
    table.add_column("Plan ID", style="dim")

    for e in entries:
        table.add_row(
            str(e.get("topic_id", "")),
            str(e.get("title", "")),
            str(e.get("game", "")),
            str(e.get("shape", "")),
            str(e.get("plan_id", "")),
        )
    console.print(table)


@topics_group.command(name="plan")
@click.argument("game_arg", required=False, default=None, metavar="[GAME]")
@click.argument("topic_arg", required=False, default=None, metavar="[TOPIC]")
@click.option("--game", "-g", default=None, help="Game name")
@click.option("--topic", "-t", default=None, help="Topic title or idea")
@click.pass_context
def topics_plan(
    ctx: click.Context,
    game_arg: str | None,
    topic_arg: str | None,
    game: str | None,
    topic: str | None,
) -> None:
    """Draft a topic plan with outline, script, clips, and validation."""
    from streamclipper.topics.planner import format_plan_outline, plan_topic_local

    resolved_game = (game or game_arg or "").strip()
    resolved_topic = (topic or topic_arg or "").strip()
    if not resolved_game or not resolved_topic:
        raise click.UsageError(
            "Missing required game or topic. Provide either '[GAME] [TOPIC]' arguments or '--game/-g' and '--topic/-t' options."
        )

    cfg: Config = ctx.obj["config"]
    with console.status(f"[bold cyan]Drafting topic plan for '{resolved_topic}'...[/bold cyan]"):
        plan = plan_topic_local(cfg, resolved_game, resolved_topic)

    console.print(format_plan_outline(plan))
    if plan.problems:
        console.print("\n[yellow]Validation Problems to Resolve:[/yellow]")
        for pb in plan.problems:
            console.print(f"  [red]•[/red] {pb}")
        console.print(f"\n[dim]Draft saved with ID [bold]{plan.plan_id}[/bold]. Edit plan.json and run 'streamclipper topics validate {plan.plan_id}'.[/dim]")
    else:
        console.print(f"\n[green]Plan is completely valid! Saved with ID [bold]{plan.plan_id}[/bold].[/green]")
        console.print(f"[dim]Run 'streamclipper topics approve {plan.plan_id}' after reviewing outline with the user.[/dim]")


@topics_group.command(name="show")
@click.argument("plan_id")
@click.pass_context
def topics_show(ctx: click.Context, plan_id: str) -> None:
    """Display rich outline, script, clips, Shorts slots, and timings for a topic plan."""
    from streamclipper.topics.planner import format_plan_outline, load_plan
    from streamclipper.topics.validate import validate_plan

    cfg: Config = ctx.obj["config"]
    plan = load_plan(cfg, plan_id)
    if not plan:
        console.print(f"[red]Topic plan '{plan_id}' not found.[/red]")
        ctx.exit(1)

    problems = validate_plan(cfg, plan)
    console.print(format_plan_outline(plan))
    if problems:
        console.print("\n[yellow]Validation Problems:[/yellow]")
        for pb in problems:
            console.print(f"  [red]•[/red] {pb}")
    else:
        console.print("\n[green]Status: Valid and ready for approval.[/green]")


@topics_group.command(name="validate")
@click.argument("plan_id")
@click.pass_context
def topics_validate(ctx: click.Context, plan_id: str) -> None:
    """Re-validate a topic plan (e.g. after editing plan.json) and update calculated timings."""
    from streamclipper.topics.discover import _format_time
    from streamclipper.topics.planner import load_plan, save_plan
    from streamclipper.topics.validate import validate_plan

    cfg: Config = ctx.obj["config"]
    plan = load_plan(cfg, plan_id)
    if not plan:
        console.print(f"[red]Topic plan '{plan_id}' not found.[/red]")
        ctx.exit(1)

    problems = validate_plan(cfg, plan)
    plan.problems = problems
    save_plan(cfg, plan)

    if problems:
        console.print(f"[yellow]Plan '{plan_id}' has {len(problems)} validation problems:[/yellow]")
        for pb in problems:
            console.print(f"  [red]•[/red] {pb}")
        ctx.exit(1)
    else:
        ratio = (plan.narration_seconds / plan.estimated_seconds) if plan.estimated_seconds > 0 else 0.0
        console.print(f"[green]Plan '{plan_id}' is valid![/green]")
        console.print(f"Estimated Length: {_format_time(plan.estimated_seconds)} ({plan.estimated_seconds:.1f}s)")
        console.print(f"Narration Length: {_format_time(plan.narration_seconds)} ({plan.narration_seconds:.1f}s, {ratio:.1%})")


@topics_group.command(name="approve")
@click.argument("plan_id")
@click.pass_context
def topics_approve(ctx: click.Context, plan_id: str) -> None:
    """Approve a valid topic plan and record it in the ledger."""
    from datetime import datetime, timezone
    from streamclipper.topics.ledger import TopicLedger
    from streamclipper.topics.planner import load_plan, save_plan
    from streamclipper.topics.validate import validate_plan

    cfg: Config = ctx.obj["config"]
    plan = load_plan(cfg, plan_id)
    if not plan:
        console.print(f"[red]Topic plan '{plan_id}' not found.[/red]")
        ctx.exit(1)

    problems = validate_plan(cfg, plan)
    if problems:
        console.print(f"[red]Cannot approve plan '{plan_id}' with validation problems:[/red]")
        for pb in problems:
            console.print(f"  [red]•[/red] {pb}")
        ctx.exit(1)

    plan.status = "approved"
    plan.approved_at = datetime.now(timezone.utc).isoformat()
    save_plan(cfg, plan)

    ledger = TopicLedger(cfg)
    ledger.record(
        topic_id=plan.topic_id,
        title=plan.topic_title or plan.video_title,
        game=plan.game,
        shape=plan.shape,
        plan_id=plan.plan_id,
    )
    console.print(f"[green]Plan '{plan_id}' approved successfully at {plan.approved_at} and recorded in ledger![/green]")


@topics_group.command(name="render")
@click.argument("plan_id")
@click.pass_context
def topics_render(ctx: click.Context, plan_id: str) -> None:
    """Render a 16:9 long topic video from an approved topic plan."""
    from streamclipper.assembly.longform import render_long_video
    from streamclipper.topics.planner import load_plan

    cfg: Config = ctx.obj["config"]
    plan = load_plan(cfg, plan_id)
    if not plan:
        console.print(f"[red]Topic plan '{plan_id}' not found.[/red]")
        ctx.exit(1)

    if plan.status != "approved":
        console.print(
            f"[red]Topic plan '{plan_id}' must be 'approved' to render (current: '{plan.status}').[/red]\n"
            f"[yellow]Run 'streamclipper topics approve {plan_id}' first after user review.[/yellow]"
        )
        ctx.exit(1)

    class CLIProgress:
        def stage(self, text: str) -> None:
            console.print(f"  [cyan]•[/cyan] {text}")

    progress = CLIProgress()
    try:
        with console.status(f"[bold cyan]Rendering topic video '{plan.topic_title}'...[/bold cyan]") as status:
            def update_stage(text: str) -> None:
                status.update(f"[bold cyan]{text}...[/bold cyan]")
                console.print(f"  [cyan]•[/cyan] {text}")

            progress.stage = update_stage
            result = render_long_video(cfg, plan, progress=progress)
    except Exception as exc:
        console.print(f"[bold red]Render failed:[/bold red] {exc}")
        ctx.exit(1)

    console.print(
        f"\n[bold green]Render completed successfully![/bold green]\n"
        f"  [bold]Video:[/bold] {result.get('video_path')}\n"
        f"  [bold]Duration:[/bold] {result.get('duration', 0.0):.1f}s (narration share: {result.get('narration_share', 0.0)*100:.1f}%)\n"
        f"  [bold]Chapters:[/bold] {result.get('chapters_path')}\n"
        f"  [bold]Thumbnail:[/bold] {result.get('thumbnail_path')}\n"
        f"  [bold]Post file:[/bold] {result.get('post_path')}\n"
    )


@topics_group.command("shorts")
@click.argument("plan_id")
@click.option(
    "--slot",
    type=click.Choice(["teaser", "short_a", "short_b"], case_sensitive=False),
    default=None,
    help="Render only a specific Short slot (teaser, short_a, short_b). Defaults to all.",
)
@click.pass_context
def topics_shorts(ctx: click.Context, plan_id: str, slot: str | None) -> None:
    """Render 9:16 vertical topic Shorts (teaser and/or standalone) from an approved topic plan."""
    from streamclipper.assembly.topic_shorts import render_topic_shorts
    from streamclipper.topics.planner import load_plan

    cfg: Config = ctx.obj["config"]
    plan = load_plan(cfg, plan_id)
    if not plan:
        console.print(f"[red]Topic plan '{plan_id}' not found.[/red]")
        ctx.exit(1)

    if plan.status != "approved":
        console.print(
            f"[red]Topic plan '{plan_id}' must be 'approved' to render (current: '{plan.status}').[/red]\n"
            f"[yellow]Run 'streamclipper topics approve {plan_id}' first after user review.[/yellow]"
        )
        ctx.exit(1)

    class CLIProgress:
        def stage(self, text: str) -> None:
            console.print(f"  [cyan]•[/cyan] {text}")

    progress = CLIProgress()
    target_slots = (slot,) if slot else ("teaser", "short_a", "short_b")
    try:
        with console.status(f"[bold cyan]Rendering topic shorts for '{plan.topic_title}'...[/bold cyan]") as status:
            def update_stage(text: str) -> None:
                status.update(f"[bold cyan]{text}...[/bold cyan]")
                console.print(f"  [cyan]•[/cyan] {text}")

            progress.stage = update_stage
            result = render_topic_shorts(cfg, plan, slots=target_slots, progress=progress)
    except Exception as exc:
        console.print(f"[bold red]Render failed:[/bold red] {exc}")
        ctx.exit(1)

    rendered = result.get("rendered") or []
    quarantined = result.get("quarantined") or []

    if rendered:
        console.print(f"\n[bold green]Rendered {len(rendered)} Short(s) successfully![/bold green]")
        for item in rendered:
            console.print(
                f"  [bold]Slot:[/bold] {item.get('slot')}\n"
                f"    [bold]Video:[/bold] {item.get('video_path')}\n"
                f"    [bold]Duration:[/bold] {item.get('duration', 0.0):.1f}s\n"
                f"    [bold]Post file:[/bold] {item.get('post_path')}"
            )

    if quarantined:
        console.print(f"\n[bold yellow]Quarantined {len(quarantined)} Short(s) (narration gate):[/bold yellow]")
        for item in quarantined:
            console.print(
                f"  [bold]Slot:[/bold] {item.get('slot')}\n"
                f"    [bold]Path:[/bold] {item.get('video_path')}\n"
                f"    [bold]Reason:[/bold] {item.get('quarantine_reason')}"
            )



if __name__ == "__main__":
    cli()
