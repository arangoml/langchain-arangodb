from typing import Dict, Tuple, Union

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


class ArangoTranslator(Visitor):
    """Translate `ArangoDB` internal query language elements to valid filters."""

    allowed_operators = [Operator.AND, Operator.OR]
    """Subset of allowed logical operators."""

    allowed_comparators = [
        Comparator.EQ,
        Comparator.NE,
        Comparator.GTE,
        Comparator.LTE,
        Comparator.LT,
        Comparator.GT,
    ]

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        map_dict = {
            Operator.AND: "AND",
            Operator.OR: "OR",
            Comparator.EQ: "==",
            Comparator.NE: "!=",
            Comparator.GTE: ">=",
            Comparator.LTE: "<=",
            Comparator.LT: "<",
            Comparator.GT: ">",
        }
        return map_dict[func]

    def visit_operation(self, operation: Operation) -> Dict:
        args = [arg.accept(self) for arg in operation.arguments]
        return {self._format_func(operation.operator): args}

    def visit_comparison(self, comparison: Comparison) -> Dict:
        return {
            comparison.attribute: {
                self._format_func(comparison.comparator): comparison.value
            }
        }

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
