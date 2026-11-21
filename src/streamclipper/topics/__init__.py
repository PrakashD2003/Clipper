"""Topic video discovery, evidence gathering, candidate scoring, and ledger management."""
from streamclipper.topics.discover import (
    evaluate_topic,
    suggest_topics_local,
    topic_material,
)
from streamclipper.topics.ledger import TopicLedger
from streamclipper.topics.models import EvidenceRef, TopicCandidate

__all__ = [
    "EvidenceRef",
    "TopicCandidate",
    "TopicLedger",
    "evaluate_topic",
    "suggest_topics_local",
    "topic_material",
]
