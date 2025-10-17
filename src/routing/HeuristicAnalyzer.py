"""Analyzes a given query in order to determine the most suitable translation methods."""
import logging

from core.types import QueryStr, HeuristicAnalysisParameters, HeuristicAnalysis

logger: logging.Logger = logging.getLogger(__name__)


class HeuristicAnalyzer:
    def __init__(self, query: QueryStr, params: HeuristicAnalysisParameters):
        self.query = query
        self.params = params

    def analyze(self) -> HeuristicAnalysis:
        logger.debug(f"Analyzing query: {self.query}")
        return HeuristicAnalysis({
            "is_question": self.query.strip().endswith("?"),
            "has_logical_operators": any(op in self.query.lower() for op in [" and ", " or ", " not "]),
            "is_comparative": any(
                comp in self.query.lower() for comp in [" better ", " worse ", " more ", " less ", " than "]),
            "is_how_to": self.query.lower().startswith("how to"),
            "is_short": len(self.query.split()) <= self.params.short_len_le,
            "is_ambiguous": any(word in self.query.lower() for word in ["maybe", "possibly", "could", "might"])
        })

    @staticmethod
    def check_format(analysis: HeuristicAnalysis) -> bool:
        required_keys = [
            "is_question",
            "has_logical_operators",
            "is_comparative",
            "is_how_to",
            "is_short",
            "is_ambiguous"
        ]
        return all(analysis.get(key) is not None for key in required_keys)
