import re
from typing import Optional, Tuple


def remove_string_literals(aql_query: str) -> str:
    """Remove string literals from the AQL query to avoid false positives.
    Args:
        aql_query: The original AQL query."""
    query_no_strings = re.sub(
        r"""
        (?:                     # Non-capturing group for alternatives
            '(?:[^'\\]|\\.)*'   # Single-quoted strings with escape handling
            |                   # OR
            "(?:[^"\\]|\\.)*"   # Double-quoted strings with escape handling
        )
    """,
        "",
        aql_query,
        flags=re.VERBOSE,
    )
    return query_no_strings.strip()


def remove_comments(aql_query: str) -> str:
    """Remove comments from the AQL query to avoid false positives.
    Args:
        aql_query: The original AQL query."""
    query_no_comments = re.sub(
        r"""
        //.*?$|/\*.*?\*/  # Single-line and multi-line comments
        """,
        "",
        aql_query,
        flags=re.DOTALL | re.MULTILINE | re.VERBOSE,
    )
    return query_no_comments.strip()


def is_aql_write_operation(aql_query: str) -> Tuple[bool, Optional[str]]:
    """Perform syntax-aware validation of AQL queries
    to accurately identify write operations.
    This function expects string literals and comments
    to be removed before checking for write operations,
    reducing false positives from keywords appearing in non-executable contexts.
    Args:
        aql_query: The AQL query to validate.
    Returns:
        str: The cleaned AQL query without string literals and comments.
    """
    dangerous_patterns = [
        r"\bINSERT\s+(?:\{[^}]*\}|\w+)\s+(?:INTO|IN)\s+\w+",
        r"\bUPDATE\s+(?:\{[^}]*\}|\w+)\s+IN\s+\w+",  # catch UPDATE syntax 1
        # catch UPDATE syntax 2
        r"\bUPDATE\s+(?:\{[^}]*\}|\w+)\s+WITH\s+\{[^}]*\}\s+IN\s+\w+",
        # catch UPDATE syntax 3 & 4
        r"\bUPDATE\s+(?:\{[^}]*\}|\w\.\w+)\s+WITH\s+\{[^}]*\}\s+IN\s+\w+",
        r"\bREPLACE\s+(?:\{[^}]*\}|\w+)\s+IN\s+\w+",  # catch REPLACE syntax 1
        # catch REPLACE syntax 2
        r"\bREPLACE\s+(?:\{[^}]*\}|\w+)\s+WITH\s+\{[^}]*\}\s+IN\s+\w+",
        # catch REPLACE syntax 3 & 4
        r"\bREPLACE\s+(?:\{[^}]*\}|\w\.\w+)\s+WITH\s+\{[^}]*\}\s+IN\s+\w+",
        # Catch REMOVE syntax 1 & 2
        r"\bREMOVE\s+(?:\{[^}]*\}|\w+)\s+IN\s+\w+",
        # Catch REMOVE syntax 3
        r"\bREMOVE\s+(?:\{[^}]*\}|\w\.\w+)\s+IN\s+\w+",
        # Catch UPSERT syntax
        r"""\bUPSERT\s+\{[^}]*\}\s+INSERT\s+\{[^}]*\}\s+
        (?:UPDATE|REPLACE)\s+\{[^}]*\}\s+IN\s+\w+""",
    ]
    for pattern in dangerous_patterns:
        result = re.search(pattern, aql_query, re.IGNORECASE | re.DOTALL)

        if result:
            return True, result.group(0)

    return False, None
