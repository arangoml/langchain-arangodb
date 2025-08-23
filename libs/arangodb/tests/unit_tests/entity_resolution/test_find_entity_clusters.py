from typing import Any
from unittest.mock import MagicMock, patch
from packaging import version

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
    mock_embedding = MagicMock()
    
    mock_db.has_collection.return_value = True
    mock_db.collection.return_value = mock_collection
    mock_db.version.return_value = "3.12.5"  # Version that supports approx search
    
    # Mock vector index
    mock_collection.indexes.return_value = [
        {
            "name": "vector_index",
            "type": "vector",
            "fields": ["embedding"],
            "id": "12345",
        }
    ]
    
    vector_store = ArangoVector(
        embedding=mock_embedding,
        embedding_dimension=64,
        database=mock_db,
        collection_name="test_collection",
        embedding_field="embedding",
        distance_strategy=DistanceStrategy.COSINE,
    )
    
    return vector_store


class TestFindEntityClusters:
    """Test cases for find_entity_clusters method."""
    
    def test_basic_clustering_default_params(self, mock_vector_store: ArangoVector) -> None:
        """Test basic entity clustering with default parameters."""
        # Mock AQL query results for main clustering
        mock_clusters = [
            {"entity": "doc1", "similar": ["doc2", "doc3"]},
            {"entity": "doc4", "similar": ["doc5"]},
        ]
        
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        result = mock_vector_store.find_entity_clusters()
        
        # Should return simple list of clusters (default behavior)
        assert result == mock_clusters
        assert len(result) == 2
        assert result[0]["entity"] == "doc1"
        assert result[0]["similar"] == ["doc2", "doc3"]
        
        # Verify AQL query was called
        mock_vector_store.db.aql.execute.assert_called()
        
    def test_clustering_with_custom_params(self, mock_vector_store: ArangoVector) -> None:
        """Test entity clustering with custom threshold and k values."""
        # Reset mock to ensure clean state
        mock_vector_store.db.aql.execute.reset_mock()
        
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        result = mock_vector_store.find_entity_clusters(
            threshold=0.9, 
            k=2, 
            use_approx=False
        )
        
        assert result == mock_clusters
        
        # Verify bind variables were passed correctly (first and only call)
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["threshold"] == 0.9
        assert bind_vars["k"] == 2
        
    def test_clustering_with_subset_relations_analysis(self, mock_vector_store: ArangoVector) -> None:
        """Test entity clustering with subset relations analysis."""
        mock_clusters = [
            {"entity": "doc1", "similar": ["doc2", "doc3"]},
            {"entity": "doc4", "similar": ["doc2", "doc3", "doc5"]},
        ]
        
        mock_subsets = [
            {"subsetGroup": "doc1", "supersetGroup": "doc4"}
        ]
        
        # Mock first call (clusters) and second call (subsets)
        mock_cursor_1 = MagicMock()
        mock_cursor_1.__iter__ = lambda self: iter(mock_clusters)
        mock_cursor_2 = MagicMock()
        mock_cursor_2.__iter__ = lambda self: iter(mock_subsets)
        
        mock_vector_store.db.aql.execute.side_effect = [mock_cursor_1, mock_cursor_2]
        
        result = mock_vector_store.find_entity_clusters(use_subset_relations=True)
        
        # Should return dictionary with both clusters and subset relationships
        assert isinstance(result, dict)
        assert "clusters" in result
        assert "subset_relationships" in result
        assert result["clusters"] == mock_clusters
        assert result["subset_relationships"] == mock_subsets
        
        # Verify both AQL queries were called
        assert mock_vector_store.db.aql.execute.call_count == 2
        
    def test_empty_results(self, mock_vector_store: ArangoVector) -> None:
        """Test behavior when no clusters are found."""
        # Mock empty results
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter([])
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Test without subset relations
        result = mock_vector_store.find_entity_clusters()
        assert result == []
        
        # Test with subset relations
        result = mock_vector_store.find_entity_clusters(use_subset_relations=True)
        assert result == {"clusters": [], "subset_relationships": []}
        
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
        
        mock_vector_store.db.aql.execute.side_effect = [mock_cursor_1, mock_cursor_2]
        
        result = mock_vector_store.find_entity_clusters(use_subset_relations=True)
        
        assert result["clusters"] == mock_clusters
        assert result["subset_relationships"] == []
        
    def test_euclidean_distance_strategy(self, mock_vector_store: ArangoVector) -> None:
        """Test entity clustering with Euclidean distance strategy."""
        # Reset mock to ensure clean state
        mock_vector_store.db.aql.execute.reset_mock()
        
        # Set distance strategy to Euclidean
        mock_vector_store._distance_strategy = DistanceStrategy.EUCLIDEAN_DISTANCE
        
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        result = mock_vector_store.find_entity_clusters(use_approx=True)
        
        # Verify query was built with Euclidean distance function (first and only call)
        call_args = mock_vector_store.db.aql.execute.call_args[0][0]
        assert "APPROX_NEAR_L2" in call_args
        assert "ASC" in call_args  # Euclidean uses ascending sort
        
    def test_cosine_distance_strategy(self, mock_vector_store: ArangoVector) -> None:
        """Test entity clustering with Cosine distance strategy."""
        # Reset mock to ensure clean state
        mock_vector_store.db.aql.execute.reset_mock()
        
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        result = mock_vector_store.find_entity_clusters(use_approx=True)
        
        # Verify query was built with Cosine distance function (first and only call)
        call_args = mock_vector_store.db.aql.execute.call_args[0][0]
        assert "APPROX_NEAR_COSINE" in call_args
        assert "DESC" in call_args  # Cosine uses descending sort
        
    def test_non_approx_search_cosine(self, mock_vector_store: ArangoVector) -> None:
        """Test entity clustering with exact search for Cosine distance."""
        # Reset mock to ensure clean state
        mock_vector_store.db.aql.execute.reset_mock()
        
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        result = mock_vector_store.find_entity_clusters(use_approx=False)
        
        # Verify query was built with exact Cosine similarity function
        call_args = mock_vector_store.db.aql.execute.call_args[0][0]
        assert "COSINE_SIMILARITY" in call_args
        assert "DESC" in call_args
        
    def test_non_approx_search_euclidean(self, mock_vector_store: ArangoVector) -> None:
        """Test entity clustering with exact search for Euclidean distance."""
        # Reset mock to ensure clean state
        mock_vector_store.db.aql.execute.reset_mock()
        
        mock_vector_store._distance_strategy = DistanceStrategy.EUCLIDEAN_DISTANCE
        
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        result = mock_vector_store.find_entity_clusters(use_approx=False)
        
        # Verify query was built with exact L2 distance function
        call_args = mock_vector_store.db.aql.execute.call_args[0][0]
        assert "L2_DISTANCE" in call_args
        assert "ASC" in call_args
        
    def test_invalid_distance_strategy(self, mock_vector_store: ArangoVector) -> None:
        """Test error handling for invalid distance strategy."""
        # Set invalid distance strategy
        mock_vector_store._distance_strategy = "INVALID_STRATEGY"
        
        with pytest.raises(ValueError) as exc_info:
            mock_vector_store.find_entity_clusters()
            
        assert "Unsupported metric" in str(exc_info.value)
        
    def test_version_check_for_approx_search(self, mock_vector_store: ArangoVector) -> None:
        """Test version check for approximate search."""
        # Mock old version that doesn't support approx search
        mock_vector_store.db.version.return_value = "3.12.3"
        
        with pytest.raises(ValueError) as exc_info:
            mock_vector_store.find_entity_clusters(use_approx=True)
            
        assert "Approximate Nearest Neighbor search requires ArangoDB >= 3.12.4" in str(exc_info.value)
        
    def test_vector_index_creation_for_approx_search(self, mock_vector_store: ArangoVector) -> None:
        """Test that vector index is created when needed for approx search."""
        # Mock no existing vector index
        mock_vector_store.collection.indexes.return_value = []
        
        with patch.object(mock_vector_store, 'retrieve_vector_index', return_value=None):
            with patch.object(mock_vector_store, 'create_vector_index') as mock_create_index:
                mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
                mock_cursor = MagicMock()
                mock_cursor.__iter__ = lambda self: iter(mock_clusters)
                mock_vector_store.db.aql.execute.return_value = mock_cursor
                
                result = mock_vector_store.find_entity_clusters(use_approx=True)
                
                # Verify vector index was created
                mock_create_index.assert_called_once()
                
    def test_aql_query_structure_basic(self, mock_vector_store: ArangoVector) -> None:
        """Test that the AQL query has the correct structure for basic clustering."""
        # Reset mock to ensure clean state
        mock_vector_store.db.aql.execute.reset_mock()
        
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        mock_vector_store.find_entity_clusters()
        
        # Get the AQL query that was executed (first and only call)
        call_args = mock_vector_store.db.aql.execute.call_args[0][0]
        
        # Verify query structure
        assert "FOR doc1 IN @@collection" in call_args
        assert "LET similar = (" in call_args
        assert "FOR doc2 IN @@collection" in call_args
        assert "FILTER score >= @threshold" in call_args
        assert "RETURN {entity: doc1._key, similar}" in call_args  
        
    def test_aql_query_structure_subsets(self, mock_vector_store: ArangoVector) -> None:
        """Test that the subset relations AQL query has the correct structure."""
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_subsets = []
        
        mock_cursor_1 = MagicMock()
        mock_cursor_1.__iter__ = lambda self: iter(mock_clusters)
        mock_cursor_2 = MagicMock()
        mock_cursor_2.__iter__ = lambda self: iter(mock_subsets)
        
        mock_vector_store.db.aql.execute.side_effect = [mock_cursor_1, mock_cursor_2]
        
        mock_vector_store.find_entity_clusters(use_subset_relations=True)
        
        # Get the subset query (second call)
        subset_query = mock_vector_store.db.aql.execute.call_args_list[1][0][0]
        
        # Verify subset query structure
        assert "FOR group1 IN @results" in subset_query
        assert "FOR group2 IN @results" in subset_query
        assert "LENGTH(group1.similar) < LENGTH(group2.similar)" in subset_query
        assert "MINUS(group1Keys, group2Keys)" in subset_query
        assert "subsetGroup: group1.entity" in subset_query
        assert "supersetGroup: group2.entity" in subset_query
        
    def test_bind_variables_passed_correctly(self, mock_vector_store: ArangoVector) -> None:
        """Test that bind variables are passed correctly to AQL queries."""
        # Reset mock to ensure clean state
        mock_vector_store.db.aql.execute.reset_mock()
        
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        threshold = 0.75
        k = 3
        
        mock_vector_store.find_entity_clusters(threshold=threshold, k=k)
        
        # Verify bind variables (first and only call)
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        
        assert bind_vars["@collection"] == "test_collection"
        assert bind_vars["threshold"] == threshold
        assert bind_vars["k"] == k
        
    def test_filter_key_clause_with_approx_search(self, mock_vector_store: ArangoVector) -> None:
        """Test filter key clause positioning with approximate search."""
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        mock_vector_store.find_entity_clusters(use_approx=True)
        
        query = mock_vector_store.db.aql.execute.call_args[0][0]
        
        # With approx search, filter should be after LIMIT
        lines = query.strip().split('\n')
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
            
    def test_filter_key_clause_without_approx_search(self, mock_vector_store: ArangoVector) -> None:
        """Test filter key clause positioning without approximate search."""
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        mock_vector_store.find_entity_clusters(use_approx=False)
        
        query = mock_vector_store.db.aql.execute.call_args[0][0]
        
        # Without approx search, filter should be before LIMIT
        lines = query.strip().split('\n')
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
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        mock_vector_store.find_entity_clusters()
        
        # Verify stream=True was passed
        call_args = mock_vector_store.db.aql.execute.call_args
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
        
        mock_vector_store.db.aql.execute.side_effect = [mock_cursor_1, mock_cursor_2]
        
        result = mock_vector_store.find_entity_clusters(use_subset_relations=True)
        
        # Verify the structure
        assert result["clusters"] == mock_clusters
        assert result["subset_relationships"] == mock_subsets
        assert len(result["clusters"]) == 3
        assert len(result["subset_relationships"]) == 1
        
    def test_edge_case_single_cluster(self, mock_vector_store: ArangoVector) -> None:
        """Test edge case with only one cluster."""
        mock_clusters = [{"entity": "doc1", "similar": ["doc2"]}]
        
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter(mock_clusters)
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        result = mock_vector_store.find_entity_clusters()
        
        assert len(result) == 1
        assert result[0]["entity"] == "doc1"
        assert result[0]["similar"] == ["doc2"]
        
    def test_edge_case_high_threshold(self, mock_vector_store: ArangoVector) -> None:
        """Test edge case with very high threshold that yields no results."""
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = lambda self: iter([])
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        result = mock_vector_store.find_entity_clusters(threshold=0.99)
        
        assert result == []
        
        # Verify high threshold was passed
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["threshold"] == 0.99


if __name__ == "__main__":
    pytest.main([__file__])
