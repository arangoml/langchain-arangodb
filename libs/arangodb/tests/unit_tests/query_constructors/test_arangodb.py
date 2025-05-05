from typing import Dict, Tuple

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)

# Import the correct translator
from langchain_arangodb.query_constructors.arangodb import ArangoTranslator

# Instantiate the translator
DEFAULT_TRANSLATOR = ArangoTranslator()

# Placeholder test cases - need to be adapted for ArangoDB's expected filter format


def test_visit_comparison() -> None:
    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=2)
    expected = {"foo": {"<": 2}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_operation() -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.LTE, attribute="abc", value=5),
        ],
    )
    expected = {
        "AND": [
            {"foo": {"<": 2}},
            {"bar": {"==": "baz"}},
            {"abc": {"<=": 5}},
        ]
    }
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual


def test_visit_structured_query() -> None:
    query = "What is the capital of France?"
    structured_query_no_filter = StructuredQuery(
        query=query,
        filter=None,
    )
    expected_no_filter: Tuple[str, Dict] = (query, {})
    actual_no_filter = DEFAULT_TRANSLATOR.visit_structured_query(
        structured_query_no_filter
    )
    assert expected_no_filter == actual_no_filter

    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=2)
    structured_query_comp = StructuredQuery(
        query=query,
        filter=comp,
    )
    expected_comp = (
        query,
        {"filter": {"foo": {"<": 2}}},
    )
    actual_comp = DEFAULT_TRANSLATOR.visit_structured_query(structured_query_comp)
    assert expected_comp == actual_comp

    op = Operation(
        operator=Operator.OR,
        arguments=[
            Comparison(comparator=Comparator.GTE, attribute="foo", value=2),
            Comparison(comparator=Comparator.NE, attribute="bar", value="baz"),
        ],
    )
    structured_query_op = StructuredQuery(
        query=query,
        filter=op,
    )
    expected_op = (
        query,
        {
            "filter": {
                "OR": [
                    {"foo": {">=": 2}},
                    {"bar": {"!=": "baz"}},
                ]
            }
        },
    )
    actual_op = DEFAULT_TRANSLATOR.visit_structured_query(structured_query_op)
    assert expected_op == actual_op 