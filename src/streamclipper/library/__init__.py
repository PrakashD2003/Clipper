"""Source library and full-text search index for topic videos."""
from streamclipper.library.frames import (
    ContactSheet,
    build_contact_sheets,
    download_proxy_video,
)
from streamclipper.library.index import LibraryIndex
from streamclipper.library.ingest import add_source, index_cached_states
from streamclipper.library.models import LibrarySource, SearchHit
from streamclipper.library.vision import (
    VisionUnavailable,
    describe_sheet_ollama,
)

__all__ = [
    "ContactSheet",
    "LibraryIndex",
    "LibrarySource",
    "SearchHit",
    "VisionUnavailable",
    "add_source",
    "build_contact_sheets",
    "describe_sheet_ollama",
    "download_proxy_video",
    "index_cached_states",
]

