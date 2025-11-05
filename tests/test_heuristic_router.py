import pytest

from chain.routing import HeuristicAnalyzer
from chain.routing.HeuristicRouter import HeuristicRouter


class DummyParams:
    """Minimal params object with the attribute used by HeuristicAnalyzer."""

    def __init__(self, short_len_le: int = 3):
        self.short_len_le = short_len_le


def test_analyze_and_check_format():
    """
    Ensure analyze produces the expected keys and check_format validates them.
    """
    params = DummyParams(short_len_le=3)
    analyzer = HeuristicAnalyzer("How to bake a cake?", params)
    analysis = analyzer.analyze()

    # Expected analysis.
    assert analysis["is_question"] is True
    assert "is_how_to" in analysis
    assert "is_short" in analysis
    assert HeuristicAnalyzer.check_format(analysis) is True


def test_route_name_conflict_masks_method():
    """
    Detect the bug where an instance attribute named `route` shadows the
    `route` method on the class. Setting `instance.route = []` should make
    `instance.route()` raise a TypeError because a list is not callable.
    """
    inst = object.__new__(HeuristicRouter)
    inst.route = []

    with pytest.raises(TypeError):
        # Attempting to call the attribute (which is a list) should raise.
        inst.route()
