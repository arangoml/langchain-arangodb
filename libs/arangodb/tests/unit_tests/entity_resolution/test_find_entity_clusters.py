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

    def test_clustering_with_subset_relations_analysis(
        self, mock_vector_store: ArangoVector
    ) -> None:
        """Test entity clustering with subset relations analysis."""
        mock_clusters = [
            {"entity": "doc1", "similar": ["doc2", "doc3"]},
            {"entity": "doc4", "similar": ["doc2", "doc3", "doc5"]},
        ]

        mock_subsets = [{"subsetGroup": "doc1", "supersetGroup": "doc4"}]

        # Mock first call (clusters) and second call (subsets)
        mock_cursor_1 = MagicMock()
        mock_cursor_1.__iter__ = lambda self: iter(mock_clusters)
        mock_cursor_2 = MagicMock()
        mock_cursor_2.__iter__ = lambda self: iter(mock_subsets)

        mock_vector_store.db.aql.execute.side_effect = [mock_cursor_1, mock_cursor_2]  # type: ignore

        result = mock_vector_store.find_entity_clusters(use_subset_relations=True)

        # Should return dictionary with both clusters and subset relationships
        assert isinstance(result, dict)
        assert "similar_entities" in result
        assert "subset_relationships" in result
        result_dict = cast(Dict[str, Any], result)
        assert result_dict["similar_entities"] == mock_clusters
        assert result_dict["subset_relationships"] == mock_subsets

        # Verify both AQL queries were called
        assert mock_vector_store.db.aql.execute.call_count == 2  # type: ignore

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

        mock_vector_store.db.aql.execute.side_effect = [mock_cursor_1, mock_cursor_2]  # type: ignore

        result = mock_vector_store.find_entity_clusters(use_subset_relations=True)

        result_dict = cast(Dict[str, Any], result)
        assert result_dict["similar_entities"] == mock_clusters
        assert result_dict["subset_relationships"] == []

    def test_euclidean_distance_strategy(self, mock_vector_store: ArangoVector) -> None:
        """Test entity clustering with Euclidean distance strategy."""
        # Reset mock to ensure clean state
        mock_vector_store.db.aql.execute.reset_mock()  # type: ignore

        # Set distance strategy to Euclidean
        mock_vector_store._distance_strategy = DistanceStrategy.EUCLIDEAN_DISTANCE

        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        mock_vector_store.find_entity_clusters(use_approx=True)

        # Verify query was built with Euclidean distance function (first and only call)
        call_args = mock_vector_store.db.aql.execute.call_args[0][0]  # type: ignore
        assert "APPROX_NEAR_L2" in call_args
        assert "ASC" in call_args  # Euclidean uses ascending sort

    def test_cosine_distance_strategy(self, mock_vector_store: ArangoVector) -> None:
        """Test entity clustering with Cosine distance strategy."""
        # Reset mock to ensure clean state
        mock_vector_store.db.aql.execute.reset_mock()  # type: ignore

        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        mock_vector_store.find_entity_clusters(use_approx=True)

        # Verify query was built with Cosine distance function (first and only call)
        call_args = mock_vector_store.db.aql.execute.call_args[0][0]  # type: ignore
        assert "APPROX_NEAR_COSINE" in call_args
        assert "DESC" in call_args  # Cosine uses descending sort

    def test_non_approx_search_cosine(self, mock_vector_store: ArangoVector) -> None:
        """Test entity clustering with exact search for Cosine distance."""
        # Reset mock to ensure clean state
        mock_vector_store.db.aql.execute.reset_mock()  # type: ignore

        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        mock_vector_store.find_entity_clusters(use_approx=False)

        # Verify query was built with exact Cosine similarity function
        call_args = mock_vector_store.db.aql.execute.call_args[0][0]  # type: ignore
        assert "COSINE_SIMILARITY" in call_args
        assert "DESC" in call_args

    def test_non_approx_search_euclidean(self, mock_vector_store: ArangoVector) -> None:
        """Test entity clustering with exact search for Euclidean distance."""
        # Reset mock to ensure clean state
        mock_vector_store.db.aql.execute.reset_mock()  # type: ignore

        mock_vector_store._distance_strategy = DistanceStrategy.EUCLIDEAN_DISTANCE

        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        mock_vector_store.find_entity_clusters(use_approx=False)

        # Verify query was built with exact L2 distance function
        call_args = mock_vector_store.db.aql.execute.call_args[0][0]  # type: ignore
        assert "L2_DISTANCE" in call_args
        assert "ASC" in call_args

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

    def test_vector_index_creation_for_approx_search(
        self, mock_vector_store: ArangoVector
    ) -> None:
        """Test that vector index is created when needed for approx search."""
        # Mock no existing vector index
        mock_vector_store.collection.indexes.return_value = []  # type: ignore

        with patch.object(
            mock_vector_store, "retrieve_vector_index", return_value=None
        ):
            with patch.object(
                mock_vector_store, "create_vector_index"
            ) as mock_create_index:
                mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
                mock_cursor = MagicMock()
                mock_cursor.__iter__ = lambda self: iter(mock_clusters)
                mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

                mock_vector_store.find_entity_clusters(use_approx=True)

                # Verify vector index was created
                mock_create_index.assert_called_once()

    def test_aql_query_structure_basic(self, mock_vector_store: ArangoVector) -> None:
        """Test that the AQL query has the correct structure for basic clustering."""
        # Reset mock to ensure clean state
        mock_vector_store.db.aql.execute.reset_mock()  # type: ignore

        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        mock_vector_store.find_entity_clusters()

        # Get the AQL query that was executed (first and only call)
        call_args = mock_vector_store.db.aql.execute.call_args[0][0]  # type: ignore

        # Verify query structure
        assert "FOR doc1 IN @@collection" in call_args
        assert "LET similar = (" in call_args
        assert "FOR doc2 IN @@collection" in call_args
        assert "FILTER score >= @threshold" in call_args
        assert "RETURN {entity: doc1._key, similar}" in call_args

    def test_aql_query_structure_subsets(self, mock_vector_store: ArangoVector) -> None:
        """Test that the subset relations AQL query has the correct structure."""
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_subsets: List[Dict[str, Any]] = []

        mock_cursor_1 = MagicMock()
        mock_cursor_1.__iter__ = lambda self: iter(mock_clusters)
        mock_cursor_2 = MagicMock()
        mock_cursor_2.__iter__ = lambda self: iter(mock_subsets)

        mock_vector_store.db.aql.execute.side_effect = [mock_cursor_1, mock_cursor_2]  # type: ignore

        mock_vector_store.find_entity_clusters(use_subset_relations=True)

        # Get the subset query (second call)
        subset_query = mock_vector_store.db.aql.execute.call_args_list[1][0][0]  # type: ignore

        # Verify subset query structure
        assert "FOR group1 IN @results" in subset_query
        assert "FOR group2 IN @results" in subset_query
        assert "LENGTH(group1.similar) < LENGTH(group2.similar)" in subset_query
        assert "MINUS(group1Keys, group2Keys)" in subset_query
        assert "subsetGroup: group1.entity" in subset_query
        assert "supersetGroup: group2.entity" in subset_query

    def test_bind_variables_passed_correctly(
        self, mock_vector_store: ArangoVector
    ) -> None:
        """Test that bind variables are passed correctly to AQL queries."""
        # Reset mock to ensure clean state
        mock_vector_store.db.aql.execute.reset_mock()  # type: ignore

        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        threshold = 0.75
        k = 3

        mock_vector_store.find_entity_clusters(threshold=threshold, k=k)

        # Verify bind variables (first and only call)
        call_args = mock_vector_store.db.aql.execute.call_args  # type: ignore
        bind_vars = call_args[1]["bind_vars"]

        assert bind_vars["@collection"] == "test_collection"
        assert bind_vars["threshold"] == threshold
        assert bind_vars["k"] == k

    def test_filter_key_clause_with_approx_search(
        self, mock_vector_store: ArangoVector
    ) -> None:
        """Test filter key clause positioning with approximate search."""
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        mock_vector_store.find_entity_clusters(use_approx=True)

        query = mock_vector_store.db.aql.execute.call_args[0][0]  # type: ignore

        # With approx search, filter should be after LIMIT
        lines = query.strip().split("\n")
        lines = [line.strip() for line in lines if line.strip()]

        # Find the positions of key elements
        limit_pos = None
        filter_pos = None

        for i, line in enumerate(lines):
            if "LIMIT @k" in line:
                limit_pos = i
            elif "FILTER doc1._key < doc2._key" in line:
                filter_pos = i

        # Filter should come after LIMIT in approx search
        if limit_pos is not None and filter_pos is not None:
            assert filter_pos > limit_pos

    def test_filter_key_clause_without_approx_search(
        self, mock_vector_store: ArangoVector
    ) -> None:
        """Test filter key clause positioning without approximate search."""
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        mock_vector_store.find_entity_clusters(use_approx=False)

        query = mock_vector_store.db.aql.execute.call_args[0][0]  # type: ignore

        # Without approx search, filter should be before LIMIT
        lines = query.strip().split("\n")
        lines = [line.strip() for line in lines if line.strip()]

        # Find the positions of key elements
        limit_pos = None
        filter_pos = None

        for i, line in enumerate(lines):
            if "LIMIT @k" in line:
                limit_pos = i
            elif "FILTER doc1._key < doc2._key" in line:
                filter_pos = i

        # Filter should come before LIMIT in non-approx search
        if limit_pos is not None and filter_pos is not None:
            assert filter_pos < limit_pos

    def test_stream_parameter_used(self, mock_vector_store: ArangoVector) -> None:
        """Test that stream=True is used in AQL execution."""
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        mock_vector_store.find_entity_clusters()

        # Verify stream=True was passed
        call_args = mock_vector_store.db.aql.execute.call_args  # type: ignore
        assert call_args[1]["stream"] is True

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

        mock_vector_store.db.aql.execute.side_effect = [mock_cursor_1, mock_cursor_2]  # type: ignore

        result = mock_vector_store.find_entity_clusters(use_subset_relations=True)

        # Verify the structure
        result_dict = cast(Dict[str, Any], result)
        assert result_dict["similar_entities"] == mock_clusters
        assert result_dict["subset_relationships"] == mock_subsets
        assert len(result_dict["similar_entities"]) == 3
        assert len(result_dict["subset_relationships"]) == 1

    def test_edge_case_single_cluster(self, mock_vector_store: ArangoVector) -> None:
        """Test edge case with only one cluster."""
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]

        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        result = mock_vector_store.find_entity_clusters()

        assert len(result) == 1
        result_list = cast(List[Dict[str, Any]], result)
        assert result_list[0]["entity"] == "doc1"
        assert result_list[0]["similar"] == ["doc2"]

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
        """Test warning when merge_similar_entities=True but use_subset_relations=False."""
        import warnings
        
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = mock_vector_store.find_entity_clusters(
                use_subset_relations=False, 
                merge_similar_entities=True
            )
            
            # Should return basic clusters and issue warning
            assert result == mock_clusters
            assert len(w) == 1
            expected_msg = ("merge_similar_entities=True requires "
                           "use_subset_relations=True")
            assert expected_msg in str(w[0].message)
            assert issubclass(w[0].category, UserWarning)

    def test_merge_entities_with_subset_relationships(
        self, mock_vector_store: ArangoVector
    ) -> None:
        """Test entity merging when subset relationships exist."""
        mock_clusters = [
            {"entity": "doc1", "similar": ["doc2", "doc3"]},
            {"entity": "doc4", "similar": ["doc2", "doc3", "doc5"]},
            {"entity": "doc6", "similar": ["doc7"]},
        ]

        mock_subsets = [
            {"subsetGroup": "doc1", "supersetGroup": "doc4"}
        ]

        mock_merged = [
            {"entity": "doc4", "merged_entities": ["doc1", "doc2", "doc3", "doc5"]},
            {"entity": "doc6", "merged_entities": ["doc7"]},
        ]

        # Mock three calls: clusters, subsets, and merge
        mock_cursor_1 = MagicMock()
        mock_cursor_1.__iter__ = lambda self: iter(mock_clusters)
        mock_cursor_2 = MagicMock()
        mock_cursor_2.__iter__ = lambda self: iter(mock_subsets)
        mock_cursor_3 = MagicMock()
        mock_cursor_3.__iter__ = lambda self: iter(mock_merged)

        mock_vector_store.db.aql.execute.side_effect = [
            mock_cursor_1, 
            mock_cursor_2, 
            mock_cursor_3
        ]  # type: ignore

        result = mock_vector_store.find_entity_clusters(
            use_subset_relations=True, 
            merge_similar_entities=True
        )

        # Should return dictionary with all three keys
        assert isinstance(result, dict)
        result_dict = cast(Dict[str, Any], result)
        assert "similar_entities" in result_dict
        assert "subset_relationships" in result_dict
        assert "merged_entities" in result_dict
        
        assert result_dict["similar_entities"] == mock_clusters
        assert result_dict["subset_relationships"] == mock_subsets
        assert result_dict["merged_entities"] == mock_merged

        # Verify all three AQL queries were called
        assert mock_vector_store.db.aql.execute.call_count == 3  # type: ignore

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

        mock_vector_store.db.aql.execute.side_effect = [mock_cursor_1, mock_cursor_2]  # type: ignore

        result = mock_vector_store.find_entity_clusters(
            use_subset_relations=True, 
            merge_similar_entities=True
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

    def test_merge_entities_empty_initial_results(
        self, mock_vector_store: ArangoVector
    ) -> None:
        """Test entity merging when no initial clusters are found."""
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter([])
        mock_vector_store.db.aql.execute.return_value = mock_cursor  # type: ignore

        result = mock_vector_store.find_entity_clusters(
            use_subset_relations=True, 
            merge_similar_entities=True
        )

        # Should return empty structure
        assert result == {"similar_entities": [], "subset_relationships": []}

        # Verify only one AQL query was called (initial clustering)
        assert mock_vector_store.db.aql.execute.call_count == 1  # type: ignore

    def test_merge_entities_aql_query_structure(
        self, mock_vector_store: ArangoVector
    ) -> None:
        """Test that the merge AQL query has the correct structure."""
        mock_clusters = [
            {"entity": "doc1", "similar": ["doc2"]},
            {"entity": "doc3", "similar": ["doc2", "doc4"]},
        ]

        mock_subsets = [{"subsetGroup": "doc1", "supersetGroup": "doc3"}]
        mock_merged = [{"entity": "doc3", "merged_entities": ["doc1", "doc2", "doc4"]}]

        mock_cursor_1 = MagicMock()
        mock_cursor_1.__iter__ = lambda self: iter(mock_clusters)
        mock_cursor_2 = MagicMock()
        mock_cursor_2.__iter__ = lambda self: iter(mock_subsets)
        mock_cursor_3 = MagicMock()
        mock_cursor_3.__iter__ = lambda self: iter(mock_merged)

        mock_vector_store.db.aql.execute.side_effect = [
            mock_cursor_1, 
            mock_cursor_2, 
            mock_cursor_3
        ]  # type: ignore

        mock_vector_store.find_entity_clusters(
            use_subset_relations=True, 
            merge_similar_entities=True
        )

        # Get the merge query (third call)
        merge_query_call = mock_vector_store.db.aql.execute.call_args_list[2]  
        merge_query = merge_query_call[0][0]

        # Verify merge query structure
        assert "FOR group IN @results" in merge_query
        assert "LET isSubset = LENGTH(" in merge_query
        assert "FILTER NOT isSubset" in merge_query
        assert "LET entitiesToMerge = (" in merge_query
        assert "UNION_DISTINCT(group.similar, entitiesToMerge)" in merge_query
        expected_return = ("RETURN { entity: group.entity, "
                          "merged_entities: mergedSimilar }")
        assert expected_return in merge_query

        # Verify bind variables for merge query
        bind_vars = merge_query_call[1]["bind_vars"]
        assert "results" in bind_vars
        assert "subsets" in bind_vars
        assert bind_vars["results"] == mock_clusters
        assert bind_vars["subsets"] == mock_subsets

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
            mock_cursor_3
        ]  # type: ignore

        result = mock_vector_store.find_entity_clusters(
            use_subset_relations=True, 
            merge_similar_entities=True
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
            cluster for cluster in result_dict["merged_entities"] 
            if cluster["entity"] == "E"
        )
        assert "A" in e_cluster["merged_entities"]  # Merged from A
        assert "C" in e_cluster["merged_entities"]  # Merged from C


if __name__ == "__main__":
    pytest.main([__file__])
