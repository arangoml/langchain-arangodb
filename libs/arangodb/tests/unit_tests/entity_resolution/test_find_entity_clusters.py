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
        """Test basic functionality of find_entity_clusters."""
        # Mock the AQL execution with sample results
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([
            {
                "entity1": {"_key": "doc1", "text": "Apple Inc.", "embedding": [0.1] * 64},
                "entity2": {"_key": "doc2", "text": "Apple Corporation", "embedding": [0.15] * 64},
                "similarity_score": 0.85
            },
            {
                "entity1": {"_key": "doc3", "text": "Microsoft Corp", "embedding": [0.2] * 64},
                "entity2": {"_key": "doc4", "text": "Microsoft Corporation", "embedding": [0.25] * 64},
                "similarity_score": 0.82
            }
        ]))
        
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Call the method
        results = mock_vector_store.find_entity_clusters(threshold=0.8)
        
        # Verify AQL query was executed with correct parameters
        mock_vector_store.db.aql.execute.assert_called_once()
        call_args = mock_vector_store.db.aql.execute.call_args
        
        # Check the query contains expected elements
        query = call_args[0][0]
        assert "FOR doc1 IN @@collection" in query
        assert "FOR doc2 IN @@collection" in query
        assert "FILTER doc1._key != doc2._key" in query
        assert "COSINE_SIMILARITY(doc1.embedding, doc2.embedding)" in query
        assert "FILTER score >= @threshold" in query
        assert "SORT score DESC" in query
        
        # Check bind variables
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["@collection"] == "test_collection"
        assert bind_vars["threshold"] == 0.8
        
        # Check stream parameter
        assert call_args[1]["stream"] is True
        
        # Verify results structure
        assert len(results) == 2
        
        # Check first result
        assert results[0]["entity1"]["_key"] == "doc1"
        assert results[0]["entity2"]["_key"] == "doc2"
        assert results[0]["similarity_score"] == 0.85
        
        # Check second result
        assert results[1]["entity1"]["_key"] == "doc3"
        assert results[1]["entity2"]["_key"] == "doc4"
        assert results[1]["similarity_score"] == 0.82

    def test_find_entity_clusters_custom_collection(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters with custom collection name."""
        # Mock empty cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Call with custom collection name
        custom_collection = "custom_entities"
        results = mock_vector_store.find_entity_clusters(
            threshold=0.9, 
            collection_name=custom_collection
        )
        
        # Verify correct collection was used in bind vars
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["@collection"] == custom_collection
        assert bind_vars["threshold"] == 0.9
        
        assert results == []

    def test_find_entity_clusters_no_results(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters when no similar documents are found."""
        # Mock empty cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Mock print to capture output
        with patch('builtins.print') as mock_print:
            results = mock_vector_store.find_entity_clusters(threshold=0.95)
        
        # Verify empty results and print statement
        assert results == []
        mock_print.assert_called_once_with(
            f"No duplicate documents found in the collection '{mock_vector_store.collection_name}'"
        )

    def test_find_entity_clusters_different_thresholds(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters with different threshold values."""
        # Mock cursor with results
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([
            {
                "entity1": {"_key": "doc1", "text": "test1"},
                "entity2": {"_key": "doc2", "text": "test2"},
                "similarity_score": 0.75
            }
        ]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Test with low threshold
        results = mock_vector_store.find_entity_clusters(threshold=0.5)
        
        # Verify threshold was passed correctly
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["threshold"] == 0.5
        
        assert len(results) == 1
        assert results[0]["similarity_score"] == 0.75

    def test_find_entity_clusters_default_parameters(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters with default parameters."""
        # Mock cursor with results
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([
            {
                "entity1": {"_key": "doc1", "text": "entity1"},
                "entity2": {"_key": "doc2", "text": "entity2"},
                "similarity_score": 0.85
            }
        ]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Call with default parameters
        results = mock_vector_store.find_entity_clusters()
        
        # Verify default threshold was used
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["threshold"] == 0.8  # default threshold
        assert bind_vars["@collection"] == mock_vector_store.collection_name
        
        assert len(results) == 1

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

    def test_find_entity_clusters_query_structure(self, mock_vector_store: ArangoVector) -> None:
        """Test that the AQL query has the correct structure."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        mock_vector_store.find_entity_clusters(threshold=0.7)
        
        # Get the executed query
        call_args = mock_vector_store.db.aql.execute.call_args
        query = call_args[0][0]
        
        # Verify query structure
        assert query.count("FOR doc1 IN @@collection") == 1
        assert query.count("FOR doc2 IN @@collection") == 1
        assert "FILTER doc1._key != doc2._key" in query
        assert "LET score = COSINE_SIMILARITY" in query
        assert "FILTER score >= @threshold" in query
        assert "SORT score DESC" in query
        assert "RETURN {" in query
        assert "entity1: doc1" in query
        assert "entity2: doc2" in query
        assert "similarity_score: score" in query

    def test_find_entity_clusters_result_processing(self, mock_vector_store: ArangoVector) -> None:
        """Test that results are processed correctly."""
        # Mock cursor with detailed results
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([
            {
                "entity1": {
                    "_key": "company1", 
                    "text": "Apple Inc.", 
                    "industry": "technology",
                    "embedding": [0.1] * 64
                },
                "entity2": {
                    "_key": "company2", 
                    "text": "Apple Corporation", 
                    "industry": "technology",
                    "embedding": [0.15] * 64
                },
                "similarity_score": 0.89
            }
        ]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        results = mock_vector_store.find_entity_clusters(threshold=0.8)
        
        # Verify result structure and content
        assert len(results) == 1
        result = results[0]
        
        # Check that all original document fields are preserved
        assert result["entity1"]["_key"] == "company1"
        assert result["entity1"]["text"] == "Apple Inc."
        assert result["entity1"]["industry"] == "technology"
        assert result["entity1"]["embedding"] == [0.1] * 64
        
        assert result["entity2"]["_key"] == "company2"
        assert result["entity2"]["text"] == "Apple Corporation"
        assert result["entity2"]["industry"] == "technology"
        assert result["entity2"]["embedding"] == [0.15] * 64
        
        assert result["similarity_score"] == 0.89

    def test_find_entity_clusters_high_threshold_no_results(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters with very high threshold that yields no results."""
        # Mock cursor with no results
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Use very high threshold
        with patch('builtins.print') as mock_print:
            results = mock_vector_store.find_entity_clusters(threshold=0.99)
        
        # Verify no results and appropriate message
        assert results == []
        mock_print.assert_called_once_with(
            f"No duplicate documents found in the collection '{mock_vector_store.collection_name}'"
        )
        
        # Verify correct threshold was used
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["threshold"] == 0.99

    def test_find_entity_clusters_error_handling(self, mock_vector_store: ArangoVector) -> None:
        """Test error handling in find_entity_clusters."""
        # Mock database error
        mock_vector_store.db.aql.execute.side_effect = Exception("Database error")
        
        # Verify that the exception is raised
        with pytest.raises(Exception, match="Database error"):
            mock_vector_store.find_entity_clusters()

    def test_find_entity_clusters_empty_collection_name(self, mock_vector_store: ArangoVector) -> None:
        """Test find_entity_clusters with empty string collection name falls back to default."""
        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_vector_store.db.aql.execute.return_value = mock_cursor
        
        # Call with empty string collection name
        results = mock_vector_store.find_entity_clusters(collection_name="")
        
        # Should fall back to default collection name since empty string is falsy
        call_args = mock_vector_store.db.aql.execute.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["@collection"] == mock_vector_store.collection_name
        
        assert results == []

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
