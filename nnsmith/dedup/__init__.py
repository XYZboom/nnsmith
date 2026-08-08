"""
nnsmith Pattern Dedup Module

A pattern-based deduplication system for nnsmith-generated programs.
Reads the same JSON pattern files as AIFuzzer's Kotlin PatternMatcher,
normalizes nnsmith's GraphIR, and matches against known bug patterns.

Usage:
    from nnsmith.dedup import DedupMatcher
    matcher = DedupMatcher(pattern_dir="...")
    if matcher.matches(gir):
        ...  # skip this program
"""

from nnsmith.dedup.matcher import DedupMatcher
from nnsmith.dedup.parser import load_patterns
from nnsmith.dedup.normalizer import normalize_graph

__all__ = ["DedupMatcher", "load_patterns", "normalize_graph"]