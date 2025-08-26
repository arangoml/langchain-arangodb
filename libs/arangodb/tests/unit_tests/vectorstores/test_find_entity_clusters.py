# type: ignore
from typing import Any, Dict, List, cast
from unittest.mock import MagicMock, patch

import pytest

from langchain_arangodb.vectorstores.arangodb_vector import (
    ArangoVector,
    DistanceStrategy,
)


@pytest.fixture
def mock_vector_store() -> ArangoVector:
    """Create a mock ArangoVector instance for testing entity clustering."""
    mock_db = MagicMock()
    mock_collection = MagicMock()

    mock_db.has_collection.return_value = True
    mock_db.collection.return_value = mock_collection
    mock_db.version.return_value = "3.12.5"  # Version that supports approx search

    # Mock AQL interface
    mock_aql = MagicMock()
    mock_db.aql = mock_aql

    # Mock vector index
    mock_collection.indexes.return_value = [
        {
            "name": "vector_index",
            "type": "vector",
            "fields": ["embedding"],
            "id": "12345",
        }
    ]

    with patch.object(ArangoVector, "__init__", lambda x, *args, **kwargs: None):
        vector_store = ArangoVector.__new__(ArangoVector)
        vector_store.db = mock_db
        vector_store.collection = mock_collection
        vector_store.collection_name = "test_collection"  # type: ignore
        vector_store.embedding_field = "embedding"  # type: ignore
        vector_store.vector_index_name = "vector_index"  # type: ignore
        vector_store._distance_strategy = DistanceStrategy.COSINE  # type: ignore

        # Mock the retrieve_vector_index method to return a valid index
        def mock_retrieve_vector_index() -> Dict[str, Any]:
            return {
                "name": "vector_index",
                "type": "vector",
                "fields": ["embedding"],
                "id": "12345",
            }

        vector_store.retrieve_vector_index = mock_retrieve_vector_index  # type: ignore

        # Mock the create_vector_index method
        vector_store.create_vector_index = MagicMock()  # type: ignore

    return vector_store


class TestFindEntityClusters:
    """Test cases for find_entity_clusters method."""

    def test_basic_clustering_default_params(
        self, mock_vector_store: ArangoVector
    ) -> None:
        """Test basic entity clustering with default parameters."""
        # Mock AQL query results for main clustering
        mock_clusters = [
            {"entity": "doc1", "similar": ["doc2", "doc3"]},
            {"entity": "doc4", "similar": ["doc5"]},
        ]

        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        result = mock_vector_store.find_entity_clusters()

        # Should return simple list of clusters (default behavior)
        assert result == mock_clusters
        assert len(result) == 2
        result_list = cast(List[Dict[str, Any]], result)
        assert result_list[0]["entity"] == "doc1"
        assert result_list[0]["similar"] == ["doc2", "doc3"]

        # Verify AQL query was called
        mock_vector_store.db.aql.execute.assert_called()  # type: ignore

    def test_clustering_with_custom_params(
        self, mock_vector_store: ArangoVector
    ) -> None:
        """Test entity clustering with custom threshold and k values."""
        # Reset mock to ensure clean state
        mock_vector_store.db.aql.execute.reset_mock()  # type: ignore

        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]

        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        result = mock_vector_store.find_entity_clusters(
            threshold=0.9, k=2, use_approx=False
        )

        assert result == mock_clusters

        # Verify bind variables were passed correctly (first and only call)
        call_args = mock_vector_store.db.aql.execute.call_args  # type: ignore
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["threshold"] == 0.9
        assert bind_vars["k"] == 2

    def test_empty_results(self, mock_vector_store: ArangoVector) -> None:
        """Test behavior when no clusters are found."""
        # Mock empty results
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter([])
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        # Test without subset relations
        result = mock_vector_store.find_entity_clusters()
        assert result == []

        # Test with subset relations
        result = mock_vector_store.find_entity_clusters(use_subset_relations=True)
        assert result == {"similar_entities": [], "subset_relationships": []}

    def test_no_subset_relations_found(self, mock_vector_store: ArangoVector) -> None:
        """Test behavior when clusters exist but no subset relations found."""
        mock_clusters = [
            {"entity": "doc1", "similar": ["doc2"]},
            {"entity": "doc3", "similar": ["doc4"]},
        ]

        # First call returns clusters, second call returns empty subsets
        mock_cursor_1 = MagicMock()
        mock_cursor_1.__iter__ = lambda self: iter(mock_clusters)
        mock_cursor_2 = MagicMock()
        mock_cursor_2.__iter__ = lambda self: iter([])

        mock_vector_store.db.aql.execute.side_effect = [mock_cursor_1, mock_cursor_2]  # type: ignore[attr-defined]

        result = mock_vector_store.find_entity_clusters(use_subset_relations=True)

        result_dict = cast(Dict[str, Any], result)
        assert result_dict["similar_entities"] == mock_clusters
        assert result_dict["subset_relationships"] == []

    def test_invalid_distance_strategy(self, mock_vector_store: ArangoVector) -> None:
        """Test error handling for invalid distance strategy."""
        # Set invalid distance strategy
        mock_vector_store._distance_strategy = "INVALID_STRATEGY"  # type: ignore

        with pytest.raises(ValueError) as exc_info:
            mock_vector_store.find_entity_clusters()

        assert "Unsupported metric" in str(exc_info.value)

    def test_version_check_for_approx_search(
        self, mock_vector_store: ArangoVector
    ) -> None:
        """Test version check for approximate search."""
        # Mock old version that doesn't support approx search
        mock_vector_store.db.version.return_value = "3.12.3"  # type: ignore

        with pytest.raises(ValueError) as exc_info:
            mock_vector_store.find_entity_clusters(use_approx=True)

        expected_msg = "ANN search requires ArangoDB >= 3.12.4"
        assert expected_msg in str(exc_info.value)

    def test_complex_clustering_scenario(self, mock_vector_store: ArangoVector) -> None:
        """Test a complex scenario with multiple clusters and subset relationships."""
        # Complex mock data with overlapping clusters
        mock_clusters = [
            {"entity": "doc1", "similar": ["doc2", "doc3"]},
            {"entity": "doc4", "similar": ["doc2", "doc3", "doc5", "doc6"]},
            {"entity": "doc7", "similar": ["doc8"]},
        ]

        mock_subsets = [
            {"subsetGroup": "doc1", "supersetGroup": "doc4"},
            # doc7 is not a subset, so it should remain
        ]

        mock_cursor_1 = MagicMock()
        mock_cursor_1.__iter__ = lambda self: iter(mock_clusters)
        mock_cursor_2 = MagicMock()
        mock_cursor_2.__iter__ = lambda self: iter(mock_subsets)

        mock_vector_store.db.aql.execute.side_effect = [mock_cursor_1, mock_cursor_2]  # type: ignore[attr-defined]

        result = mock_vector_store.find_entity_clusters(use_subset_relations=True)

        # Verify the structure
        result_dict = cast(Dict[str, Any], result)
        assert result_dict["similar_entities"] == mock_clusters
        assert result_dict["subset_relationships"] == mock_subsets
        assert len(result_dict["similar_entities"]) == 3
        assert len(result_dict["subset_relationships"]) == 1

    def test_edge_case_high_threshold(self, mock_vector_store: ArangoVector) -> None:
        """Test edge case with very high threshold that yields no results."""
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter([])
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        result = mock_vector_store.find_entity_clusters(threshold=0.99)

        assert result == []

        # Verify high threshold was passed
        call_args = mock_vector_store.db.aql.execute.call_args  # type: ignore
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["threshold"] == 0.99

    def test_merge_entities_warning_when_subset_relations_false(
        self, mock_vector_store: ArangoVector
    ) -> None:
        """Test warning when merge_similar_entities=True but
        use_subset_relations=False."""
        import warnings

        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = mock_vector_store.find_entity_clusters(
                use_subset_relations=False, merge_similar_entities=True
            )

            # Should return basic clusters and issue warning
            assert result == mock_clusters
            assert len(w) == 1
            expected_msg = (
                "merge_similar_entities=True requires use_subset_relations=True"
            )
            assert expected_msg in str(w[0].message)
            assert issubclass(w[0].category, UserWarning)

    def test_merge_entities_no_subset_relationships(
        self, mock_vector_store: ArangoVector
    ) -> None:
        """Test entity merging when no subset relationships exist."""
        mock_clusters = [
            {"entity": "doc1", "similar": ["doc2"]},
            {"entity": "doc3", "similar": ["doc4"]},
        ]

        # First call returns clusters, second call returns empty subsets
        mock_cursor_1 = MagicMock()
        mock_cursor_1.__iter__ = lambda self: iter(mock_clusters)
        mock_cursor_2 = MagicMock()
        mock_cursor_2.__iter__ = lambda self: iter([])

        mock_vector_store.db.aql.execute.side_effect = [mock_cursor_1, mock_cursor_2]  # type: ignore[attr-defined]

        result = mock_vector_store.find_entity_clusters(
            use_subset_relations=True, merge_similar_entities=True
        )

        # Should return dictionary indicating no merging was performed
        assert isinstance(result, dict)
        result_dict = cast(Dict[str, Any], result)
        assert "similar_entities" in result_dict
        assert "subset_relationships" in result_dict
        assert "merged_entities" in result_dict

        assert result_dict["similar_entities"] == mock_clusters
        assert result_dict["subset_relationships"] == []
        assert result_dict["merged_entities"] == []  # Empty - no merging occurred

        # Verify only two AQL queries were called (no merge query)
        assert mock_vector_store.db.aql.execute.call_count == 2  # type: ignore

    def test_merge_entities_complex_hierarchy(
        self, mock_vector_store: ArangoVector
    ) -> None:
        """Test entity merging with complex hierarchical subset relationships."""
        mock_clusters = [
            {"entity": "A", "similar": ["B"]},
            {"entity": "C", "similar": ["B", "D"]},
            {"entity": "E", "similar": ["B", "D", "F", "G"]},
            {"entity": "H", "similar": ["I"]},  # Standalone cluster
        ]

        mock_subsets = [
            {"subsetGroup": "A", "supersetGroup": "C"},
            {"subsetGroup": "C", "supersetGroup": "E"},
            # A is subset of C, C is subset of E, H is standalone
        ]

        mock_merged = [
            {"entity": "E", "merged_entities": ["A", "C", "B", "D", "F", "G"]},
            {"entity": "H", "merged_entities": ["I"]},
        ]

        mock_cursor_1 = MagicMock()
        mock_cursor_1.__iter__ = lambda self: iter(mock_clusters)
        mock_cursor_2 = MagicMock()
        mock_cursor_2.__iter__ = lambda self: iter(mock_subsets)
        mock_cursor_3 = MagicMock()
        mock_cursor_3.__iter__ = lambda self: iter(mock_merged)

        mock_vector_store.db.aql.execute.side_effect = [
            mock_cursor_1,
            mock_cursor_2,
            mock_cursor_3,
        ]  # type: ignore[attr-defined]

        result = mock_vector_store.find_entity_clusters(
            use_subset_relations=True, merge_similar_entities=True
        )

        result_dict = cast(Dict[str, Any], result)

        # Verify original data is preserved
        assert result_dict["similar_entities"] == mock_clusters
        assert result_dict["subset_relationships"] == mock_subsets

        # Verify merging result
        assert result_dict["merged_entities"] == mock_merged
        assert len(result_dict["merged_entities"]) == 2  # Only non-subset entities

        # Verify E (top-level entity) contains all merged entities
        e_cluster = next(
            cluster
            for cluster in result_dict["merged_entities"]
            if cluster["entity"] == "E"
        )
        assert "A" in e_cluster["merged_entities"]  # Merged from A
        assert "C" in e_cluster["merged_entities"]  # Merged from C


if __name__ == "__main__":
    pytest.main([__file__])
