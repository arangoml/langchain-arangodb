from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_arangodb.vectorstores.arangodb_vector import ArangoVector


@pytest.fixture
def mock_vector_store() -> ArangoVector:
    """Create a mock ArangoVector instance for testing."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_async_db = MagicMock()

    mock_db.has_collection.return_value = True
    mock_db.collection.return_value = mock_collection
    mock_db.begin_async_execution.return_value = mock_async_db

    with patch(
        "langchain_arangodb.vectorstores.arangodb_vector.StandardDatabase",
        return_value=mock_db,
    ):
        vector_store = ArangoVector(
            embedding=MagicMock(),
            embedding_dimension=64,
            database=mock_db,
            collection_name="test_collection",
            embedding_field="embedding",
        )

        return vector_store


class TestFindEntityClusters:
    """Test cases for the find_entity_clusters method."""

    def test_find_entity_clusters_basic_functionality(self, mock_vector_store: ArangoVector) -> None:
        """Test basic functionality of find_entity_clusters with grouped results."""
        # Mock the AQL execution with sample grouped results
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([
            {
                "entity": "doc1",
                "similar": ["doc2", "doc3", "doc4"]
            },
            {
                "entity": "doc5", 
                "similar": ["doc6", "doc7"]
            }
        ]))
        
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Call the method
        results = mock_vector_store.find_entity_clusters(threshold=0.8, k=4)
        
        # Verify AQL query was executed with correct parameters
        mock_vector_store.db.aql.execute.assert_called_once()
        call_args = mock_vector_store.db.aql.execute.call_args
        
        # Check the query contains expected elements for grouped approach
        query = call_args[0][0]
        assert "FOR doc1 IN @@collection" in query
        assert "LET similar = (" in query
        assert "FOR doc2 IN @@collection" in query
        assert "FILTER doc1._key < doc2._key" in query  # Duplicate elimination
        assert "COSINE_SIMILARITY(doc1.embedding, doc2.embedding)" in query
        assert "FILTER score >= @threshold" in query
        assert "SORT score DESC" in query
        assert "LIMIT @k" in query
        assert "RETURN doc2._key" in query
        assert "FILTER LENGTH(similar) > 0" in query
        assert "RETURN {entity: doc1._key, similar}" in query
        
        # Check bind variables
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["@collection"] == "test_collection"
        assert bind_vars["threshold"] == 0.8
        assert bind_vars["k"] == 4
        
        # Check stream parameter
        assert call_args[1]["stream"] is True
        
        # Verify results structure
        assert len(results) == 2
        
        # Check first result
        assert results[0]["entity"] == "doc1"
        assert results[0]["similar"] == ["doc2", "doc3", "doc4"]
        
        # Check second result
        assert results[1]["entity"] == "doc5"
        assert results[1]["similar"] == ["doc6", "doc7"]

    def test_find_entity_clusters_default_parameters(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters with default parameters."""
        # Mock cursor with results
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([
            {
                "entity": "doc1",
                "similar": ["doc2", "doc3"]
            }
        ]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Call with default parameters
        results = mock_vector_store.find_entity_clusters()
        
        # Verify default values were used
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["threshold"] == 0.8  # default threshold
        assert bind_vars["k"] == 4  # default k
        assert bind_vars["@collection"] == mock_vector_store.collection_name
        
        # Verify default similarity function (COSINE_SIMILARITY)
        query = call_args[0][0]
        assert "COSINE_SIMILARITY" in query
        assert "SORT score DESC" in query
        assert "FILTER score >= @threshold" in query
        
        assert len(results) == 1
        assert results[0]["entity"] == "doc1"
        assert results[0]["similar"] == ["doc2", "doc3"]

    def test_find_entity_clusters_duplicate_elimination(self, mock_vector_store: ArangoVector) -> None:
        """Test that duplicate pairs are eliminated using doc1._key < doc2._key."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        mock_vector_store.find_entity_clusters()
        
        # Verify query uses < instead of != to eliminate duplicates
        call_args = mock_vector_store.db.aql.execute.call_args
        query = call_args[0][0]
        assert "FILTER doc1._key < doc2._key" in query
        assert "FILTER doc1._key != doc2._key" not in query

    def test_find_entity_clusters_cosine_similarity_function(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters with COSINE_SIMILARITY function."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([
            {
                "entity": "doc1",
                "similar": ["doc2", "doc3"]
            }
        ]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        mock_vector_store.find_entity_clusters(similarity_function="COSINE_SIMILARITY")
        
        # Verify COSINE_SIMILARITY function is used in query
        call_args = mock_vector_store.db.aql.execute.call_args
        query = call_args[0][0]
        assert "COSINE_SIMILARITY" in query
        assert "SORT score DESC" in query
        assert "FILTER score >= @threshold" in query

    def test_find_entity_clusters_l2_distance_function(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters with L2_DISTANCE similarity function."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([
            {
                "entity": "doc1", 
                "similar": ["doc2"]
            }
        ]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        mock_vector_store.find_entity_clusters(similarity_function="L2_DISTANCE")
        
        # Verify L2_DISTANCE function is used in query
        call_args = mock_vector_store.db.aql.execute.call_args
        query = call_args[0][0]
        assert "L2_DISTANCE" in query
        assert "SORT score ASC" in query
        assert "FILTER score <= @threshold" in query

    def test_find_entity_clusters_custom_parameters(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters with custom threshold and k values."""
        # Mock cursor with results
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([
            {
                "entity": "doc1",
                "similar": ["doc2"]
            }
        ]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Call with custom parameters
        results = mock_vector_store.find_entity_clusters(threshold=0.9, k=1)
        
        # Verify custom values were used
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["threshold"] == 0.9
        assert bind_vars["k"] == 1
        
        assert len(results) == 1
        assert results[0]["entity"] == "doc1"
        assert results[0]["similar"] == ["doc2"]

    def test_find_entity_clusters_no_results(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters when no similar documents are found."""
        # Mock empty cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        results = mock_vector_store.find_entity_clusters(threshold=0.95)
        
        # Verify empty results
        assert results == []

    def test_find_entity_clusters_empty_cursor_returns_empty_list(self, mock_vector_store: ArangoVector) -> None:
        """Test that empty cursor properly returns empty list."""
        # Mock empty cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        results = mock_vector_store.find_entity_clusters()
        
        # Should return empty list, not None
        assert results == []
        assert isinstance(results, list)

    def test_find_entity_clusters_invalid_similarity_function(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters with invalid similarity function raises error."""
        with pytest.raises(ValueError, match="Unsupported similarity function: INVALID"):
            mock_vector_store.find_entity_clusters(similarity_function="INVALID")

    def test_find_entity_clusters_case_insensitive_similarity_function(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters with case-insensitive similarity function names."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Test lowercase cosine
        mock_vector_store.find_entity_clusters(similarity_function="cosine")
        call_args = mock_vector_store.db.aql.execute.call_args
        query = call_args[0][0]
        assert "COSINE_SIMILARITY" in query
        
        # Reset mock
        mock_vector_store.db.aql.execute.reset_mock()
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Test mixed case l2_distance
        mock_vector_store.find_entity_clusters(similarity_function="l2_distance")
        call_args = mock_vector_store.db.aql.execute.call_args
        query = call_args[0][0]
        assert "L2_DISTANCE" in query

    def test_find_entity_clusters_all_similarity_functions(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters with all supported similarity functions."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        similarity_functions = [
            ("COSINE_SIMILARITY", "COSINE_SIMILARITY", "DESC", ">="),
            ("L2_DISTANCE", "L2_DISTANCE", "ASC", "<="),
        ]
        
        for func_name, expected_func, expected_sort, expected_filter in similarity_functions:
            # Reset mock
            mock_vector_store.db.aql.execute.reset_mock()
            mock_vector_store.db.aql.execute.return_value = mock_cursor
            
            mock_vector_store.find_entity_clusters(similarity_function=func_name)
            
            call_args = mock_vector_store.db.aql.execute.call_args
            query = call_args[0][0]
            assert expected_func in query
            assert f"SORT score {expected_sort}" in query
            assert f"FILTER score {expected_filter} @threshold" in query

    def test_find_entity_clusters_embedding_field_usage(self, mock_vector_store: ArangoVector) -> None:
        """Test that find_entity_clusters uses the correct embedding field."""
        # Create vector store with custom embedding field
        mock_vector_store.embedding_field = "custom_embedding"
        
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        mock_vector_store.find_entity_clusters()
        
        # Verify query uses the custom embedding field
        call_args = mock_vector_store.db.aql.execute.call_args
        query = call_args[0][0]
        assert "COSINE_SIMILARITY(doc1.custom_embedding, doc2.custom_embedding)" in query

    def test_find_entity_clusters_query_structure_components(self, mock_vector_store: ArangoVector) -> None:
        """Test that the AQL query contains all required structural components."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        mock_vector_store.find_entity_clusters()
        
        # Get the executed query
        call_args = mock_vector_store.db.aql.execute.call_args
        query = call_args[0][0]
        
        # Verify all essential AQL components are present
        required_components = [
            "FOR doc1 IN @@collection",
            "LET similar = (",
            "FOR doc2 IN @@collection",
            "FILTER doc1._key < doc2._key",  # Duplicate elimination
            "LET score = COSINE_SIMILARITY",
            "SORT score DESC",
            "LIMIT @k",
            "FILTER score >= @threshold",
            "RETURN doc2._key",
            ")",  # End of subquery
            "FILTER LENGTH(similar) > 0",
            "RETURN {entity: doc1._key, similar}"
        ]
        
        for component in required_components:
            assert component in query, f"Missing component: {component}"

    def test_find_entity_clusters_stream_parameter(self, mock_vector_store: ArangoVector) -> None:
        """Test that find_entity_clusters uses stream=True for AQL execution."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        mock_vector_store.find_entity_clusters()
        
        # Verify stream=True was passed
        call_args = mock_vector_store.db.aql.execute.call_args
        assert call_args[1]["stream"] is True

    def test_find_entity_clusters_bind_vars_completeness(self, mock_vector_store: ArangoVector) -> None:
        """Test that all required bind variables are present."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        mock_vector_store.find_entity_clusters(threshold=0.85, k=3)
        
        # Verify all required bind variables are present
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        
        # Check all required bind variables
        assert "@collection" in bind_vars
        assert "threshold" in bind_vars  
        assert "k" in bind_vars
        
        # Check values
        assert bind_vars["@collection"] == "test_collection"
        assert bind_vars["threshold"] == 0.85
        assert bind_vars["k"] == 3

    def test_find_entity_clusters_threshold_edge_cases(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters with edge case threshold values."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Test with threshold = 0.0
        mock_vector_store.find_entity_clusters(threshold=0.0)
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["threshold"] == 0.0
        
        # Reset mock
        mock_vector_store.db.aql.execute.reset_mock()
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Test with threshold = 1.0
        mock_vector_store.find_entity_clusters(threshold=1.0)
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["threshold"] == 1.0

    def test_find_entity_clusters_k_edge_cases(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters with edge case k values."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Test with k = 1
        mock_vector_store.find_entity_clusters(k=1)
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["k"] == 1
        
        # Reset mock
        mock_vector_store.db.aql.execute.reset_mock()
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Test with k = 1000
        mock_vector_store.find_entity_clusters(k=1000)
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["k"] == 1000

    def test_find_entity_clusters_return_format_validation(self, mock_vector_store: ArangoVector) -> None:
        """Test that returned results have the correct format."""
        # Mock cursor with specific result format
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([
            {
                "entity": "entity_001",
                "similar": ["entity_002", "entity_003", "entity_004"]
            },
            {
                "entity": "entity_005",
                "similar": ["entity_006"]
            }
        ]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        results = mock_vector_store.find_entity_clusters()
        
        # Verify return format
        assert isinstance(results, list)
        assert len(results) == 2
        
        for result in results:
            assert isinstance(result, dict)
            assert "entity" in result
            assert "similar" in result
            assert isinstance(result["entity"], str)
            assert isinstance(result["similar"], list)
            
            # All similar entities should be strings
            for similar_entity in result["similar"]:
                assert isinstance(similar_entity, str)

    def test_find_entity_clusters_large_k_value(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters with very large k value."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([
            {
                "entity": "doc1",
                "similar": ["doc" + str(i) for i in range(2, 102)]  # 100 similar entities
            }
        ]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        results = mock_vector_store.find_entity_clusters(k=100)
        
        # Verify large k is handled correctly
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["k"] == 100
        
        # Should return all similar entities up to k limit
        assert len(results) == 1
        assert len(results[0]["similar"]) == 100

    def test_find_entity_clusters_error_handling(self, mock_vector_store: ArangoVector) -> None:
        """Test error handling in find_entity_clusters."""
        # Mock database error
        mock_vector_store.db.aql.execute.side_effect = Exception("Database error")
        
        # Verify that the exception is raised
        with pytest.raises(Exception, match="Database error"):
            mock_vector_store.find_entity_clusters()

    def test_find_entity_clusters_collection_name_usage(self, mock_vector_store: ArangoVector) -> None:
        """Test that find_entity_clusters uses the instance collection name."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Set custom collection name
        mock_vector_store.collection_name = "custom_entities"
        
        mock_vector_store.find_entity_clusters()
        
        # Verify instance collection name was used
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["@collection"] == "custom_entities"

    def test_find_entity_clusters_mixed_case_similarity_functions(self, mock_vector_store: ArangoVector) -> None:
        """Test all variations of similarity function names."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        test_cases = [
            ("cosine_similarity", "COSINE_SIMILARITY"),
            ("COSINE_SIMILARITY", "COSINE_SIMILARITY"),
            ("cosine", "COSINE_SIMILARITY"),
            ("COSINE", "COSINE_SIMILARITY"),
            ("l2_distance", "L2_DISTANCE"),
            ("L2_DISTANCE", "L2_DISTANCE"),
            ("l2", "L2_DISTANCE"),
            ("L2", "L2_DISTANCE"),
        ]
        
        for input_func, expected_func in test_cases:
            # Reset mock
            mock_vector_store.db.aql.execute.reset_mock()
            mock_vector_store.db.aql.execute.return_value = mock_cursor
            
            mock_vector_store.find_entity_clusters(similarity_function=input_func)
            
            call_args = mock_vector_store.db.aql.execute.call_args
            query = call_args[0][0]
            assert expected_func in query, f"Failed for input: {input_func}, expected: {expected_func}"

    def test_find_entity_clusters_performance_parameters(self, mock_vector_store: ArangoVector) -> None:
        """Test that performance-related parameters are correctly set."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        mock_vector_store.find_entity_clusters()
        
        # Verify stream=True is used for performance
        call_args = mock_vector_store.db.aql.execute.call_args
        kwargs = call_args[1]
        assert kwargs["stream"] is True
        
        # Verify bind_vars are used (not string interpolation for security)
        assert "bind_vars" in kwargs
        bind_vars = kwargs["bind_vars"]
        assert "@collection" in bind_vars
        assert "threshold" in bind_vars
        assert "k" in bind_vars
