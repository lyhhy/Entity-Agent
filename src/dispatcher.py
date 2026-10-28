"""
StenRAG Dispatcher
==================

Top-level orchestrator — thin wrapper around `StenGraphPipeline`.

    from src.dispatcher import build_dispatcher
    disp = build_dispatcher(graph_builder)
    result = disp.run("Which film shares more cast members with The Matrix?")
    print(result.final_answer)  # plain text, top-5 sentences

The dispatcher:
  1. Calls QueryAnalyzer on the query (for entity extraction in Step 1).
  2. Calls StenGraphPipeline.retrieve(query) — all 10 steps happen here.
  3. Returns a DispatchResult whose `final_answer` is the pipeline's
     `final_text` (plain text, top-5 sentence contents).

No DFS, no graph traversal, no constrained path search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .query_analyzer import make_query_analyzer
from .hyper_edge import HyperEdgeRegistry
from .container_manager import ContainerManager
from .stengraph_pipeline import StenGraphPipeline
from .llm_entity_extractor import make_llm_entity_extractor


@dataclass
class DispatchResult:
    path: str = ""  # always "pipeline" now
    analysis: Optional[Any] = None
    final_answer: str = ""
    pipeline_result: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "path": self.path,
            "final_answer": self.final_answer,
        }
        if self.analysis is not None:
            out["analysis"] = getattr(self.analysis, "to_dict", lambda: {})()
        if self.pipeline_result is not None:
            out["pipeline"] = self.pipeline_result.to_dict()
        return out


def build_dispatcher(
    graph_builder,
    client=None,
    nlp=None,
    hyperedge_registry: Optional[HyperEdgeRegistry] = None,
    sentence_embedder=None,
    use_hyperedge: bool = True,  # kept for compat; no-op
    strict_entity_match: bool = False,
    attr_soft_match: bool = True,  # symmetric-attribute soft matching in rerank
    fuzzy_match: bool = True,  # full fuzzy entity→container matching
    fuzzy_threshold: float = 0.58,
) -> "Dispatcher":
    """
    Wire up the QueryAnalyzer and StenGraphPipeline.
    """
    cm: ContainerManager = graph_builder.container_manager
    he = hyperedge_registry or HyperEdgeRegistry(container_manager=cm)
    analyzer = make_query_analyzer(client=client)
    # NEW: dedicated LLM-backed named-anchor extractor.  When the LLM
    # is disabled (no key), it returns [] and the analyzer path takes
    # over.  The shared cache keeps cost to ~1 cent per 100 questions.
    entity_extractor = make_llm_entity_extractor(client=client)

    pipeline = StenGraphPipeline(
        graph=graph_builder,
        container_manager=cm,
        hyperedge_registry=he,
        sentence_registry=graph_builder.sentence_registry,
        edge_registry=getattr(graph_builder, "edge_registry", None),
        analyzer=analyzer,
        nlp=nlp,
        entity_extractor=entity_extractor,
        strict_entity_match=strict_entity_match,
        fuzzy_match=fuzzy_match,
        fuzzy_threshold=fuzzy_threshold,
    )
    # Symmetric-attribute soft matching: flip the per-pipeline flag.
    pipeline.attr_soft_match_mode = attr_soft_match

    return Dispatcher(
        graph=graph_builder,
        analyzer=analyzer,
        pipeline=pipeline,
    )


class Dispatcher:
    """
    Thin dispatcher: just calls StenGraphPipeline.retrieve().
    """

    def __init__(self, graph, analyzer, pipeline: StenGraphPipeline) -> None:
        self.graph = graph
        self.analyzer = analyzer
        self.pipeline = pipeline

    def run(
        self,
        query: str,
        top_k: int = 5,
        bridge_titles: Optional[Dict[str, List[str]]] = None,
    ) -> DispatchResult:
        analysis = self.analyzer(query)
        p_result = self.pipeline.retrieve(
            query, top_k=top_k, bridge_titles=bridge_titles,
        )
        return DispatchResult(
            path="pipeline",
            analysis=analysis,
            final_answer=p_result.final_text,
            pipeline_result=p_result,
        )
