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
    type=click.Choice(["auto", "narrative", "story", "raw"], case_sensitive=False),
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
    if force:
        call_kwargs["force"] = True

    clips = asyncio.run(pipeline.process_stream(**call_kwargs))

    if not clips:
        console.print("[yellow]No clips were generated for this stream.[/]")
        return

    console.print(f"[bold green]✅ Generated {len(clips)} clip(s):[/]")
    for idx, result in enumerate(clips, 1):
        console.print(
            f"  {idx}. [cyan]{result.video_path}[/] "
            f"([dim]{result.metadata.transformation_mode}, {result.duration_seconds:.1f}s[/])"
        )


@cli.command()
@click.option("--game", "-g", default=None, help="Game to search for")
@click.option("--max-streams", "-n", default=None, type=int, help="Number of streams")
@click.option("--max-clips", default=None, type=int, help="Clips per stream")
@click.option(
    "--mode",
    type=click.Choice(["auto", "narrative", "story", "raw"], case_sensitive=False),
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


if __name__ == "__main__":
    cli()
