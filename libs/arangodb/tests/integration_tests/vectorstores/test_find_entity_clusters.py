"""
Integration tests for find_entity_clusters functionality.

These tests require:
- Running ArangoDB instance
- Real database operations with actual collections
- Comprehensive testing of clustering behavior
"""

from typing import Any, Dict, List

import pytest
from arango.database import StandardDatabase

from langchain_arangodb.vectorstores.arangodb_vector import (
    ArangoVector,
    DistanceStrategy,
)
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

EMBEDDING_DIMENSION = 384  # More realistic embedding size


@pytest.mark.usefixtures("clear_arangodb_database")
class TestFindEntityClustersIntegration:
    """Integration tests for find_entity_clusters method."""

    @pytest.fixture
    def character_documents(self) -> List[Dict[str, Any]]:
        """Create structured character documents for testing entity clustering."""
        return [
            {
                "_key": "001",
                "name": "Ned",
                "surname": "Stark",
                "house": "Stark",
                "region": "North",
                "alive": False,
                "age": 41,
                "titles": ["Lord of Winterfell", "Warden of the North"],
                "traits": ["honorable", "duty-bound", "just"],
                "text": "Ned Stark honorable lord Winterfell northern warden dutiful",
                "profession": "lord",
            },
            {
                "_key": "002",
                "name": "Jon",
                "surname": "Snow",
                "house": "Stark",
                "region": "North",
                "alive": True,
                "age": 16,
                "titles": ["Bastard of Winterfell", "Lord Commander"],
                "traits": ["honorable", "brave", "bastard"],
                "text": "Jon Snow bastard Stark honorable brave lord commander north",
                "profession": "commander",
            },
            {
                "_key": "003",
                "name": "Robb",
                "surname": "Stark",
                "house": "Stark",
                "region": "North",
                "alive": False,
                "age": 16,
                "titles": ["King in the North", "Young Wolf"],
                "traits": ["brave", "leader", "young"],
                "text": "Robb Stark young wolf king north brave leader stark heir",
                "profession": "king",
            },
            {
                "_key": "004",
                "name": "Tyrion",
                "surname": "Lannister",
                "house": "Lannister",
                "region": "Westerlands",
                "alive": True,
                "age": 32,
                "titles": ["Hand of the King", "Imp"],
                "traits": ["clever", "witty", "dwarf"],
                "text": "Tyrion Lannister clever imp witty hand king dwarf intelligent",
                "profession": "advisor",
            },
            {
                "_key": "005",
                "name": "Jaime",
                "surname": "Lannister",
                "house": "Lannister",
                "region": "Westerlands",
                "alive": True,
                "age": 35,
                "titles": ["Kingslayer", "Ser"],
                "traits": ["skilled", "golden", "knight"],
                "text": "Jaime Lannister kingslayer golden knight skilled twin",
                "profession": "knight",
            },
            {
                "_key": "006",
                "name": "Cersei",
                "surname": "Lannister",
                "house": "Lannister",
                "region": "Westerlands",
                "alive": False,
                "age": 35,
                "titles": ["Queen Regent", "Queen Mother"],
                "traits": ["ruthless", "proud", "ambitious"],
                "text": "Cersei Lannister queen regent ruthless ambitious proud mother",
                "profession": "queen",
            },
            {
                "_key": "007",
                "name": "Ceresei",
                "surname": "L",
                "house": "Lannister",
                "region": "Westerlands",
                "alive": False,
                "age": 35,
                "titles": ["Queen Regent", "Queen Mother"],
                "traits": ["ruthless", "proud", "ambitious"],
                "text": "Cersei Lannister queen regent ruthless ambitious proud mother",
                "profession": "queen",
            },
            {
                "_key": "008",
                "name": "Arya",
                "surname": "Stark",
                "house": "Stark",
                "region": "North",
                "alive": True,
                "age": 11,
                "titles": ["No One", "Faceless"],
                "traits": ["skilled", "assassin", "young"],
                "text": "Arya Stark no one faceless assassin skilled young northern",
                "profession": "assassin",
            },
            {
                "_key": "009",
                "name": "Ned Stark",
                "surname": "S",
                "house": "Stark",
                "region": "North",
                "alive": False,
                "age": 41,
                "titles": ["Lord of Winterfell", "Warden of the North"],
                "traits": ["honorable", "duty-bound", "just"],
                "text": "Ned Stark honorable lord Winterfell northern warden dutiful",
                "profession": "lord",
            },
        ]

    @pytest.fixture
    def vector_store_with_data(
        self, character_documents: List[Dict[str, Any]], db: StandardDatabase
    ) -> Any:
        """Create vector store with character data for testing."""
        collection_name = "GameOfThrones"

        # Clean up any existing collection
        if db.has_collection(collection_name):
            db.delete_collection(collection_name)

        # Create collection
        db.create_collection(collection_name)

        # Create embeddings - use deterministic fake embeddings for consistent tests
        fake_embeddings = FakeEmbeddings(dimension=EMBEDDING_DIMENSION)

        # Create vector store
        vector_store = ArangoVector(
            embedding=fake_embeddings,
            embedding_dimension=EMBEDDING_DIMENSION,
            database=db,
            collection_name=collection_name,
            text_field="text",
            embedding_field="embedding",
        )

        # Add documents with embeddings
        texts = [doc["text"] for doc in character_documents]
        metadatas = [
            {k: v for k, v in doc.items() if k not in ["_key", "text"]}
            for doc in character_documents
        ]
        ids = [doc["_key"] for doc in character_documents]

        vector_store.add_texts(
            texts=texts,  # type: ignore[arg-type]
            metadatas=metadatas,
            ids=ids,  # type: ignore[arg-type]
        )

        yield vector_store

        # Cleanup
        if db.has_collection(collection_name):
            db.delete_collection(collection_name)

    def test_basic_entity_clustering(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test basic entity clustering functionality."""
        result = vector_store_with_data.find_entity_clusters(
            threshold=0.8,  # Low threshold to ensure we get some results
            k=3,
            use_approx=False,
            use_subset_relations=False,
        )

        # Should return a list of cluster dictionaries
        assert isinstance(result, list)
        assert len(result) > 0

        # Each cluster should have the expected structure
        for cluster in result:
            assert isinstance(cluster, dict)
            assert "entity" in cluster
            assert "similar" in cluster
            assert isinstance(cluster["entity"], str)
            assert isinstance(cluster["similar"], list)

    def test_entity_clustering_with_subset_relations(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test entity clustering with subset relationship analysis."""
        result = vector_store_with_data.find_entity_clusters(
            threshold=0.7, k=4, use_approx=False, use_subset_relations=True
        )

        # Should return a dictionary with similar_entities and subset_relationships
        assert isinstance(result, dict)
        assert "similar_entities" in result
        assert "subset_relationships" in result

        clusters = result["similar_entities"]
        subsets = result["subset_relationships"]

        assert isinstance(clusters, list)
        assert isinstance(subsets, list)
        assert len(clusters) > 0

        # Validate cluster structure
        for cluster in clusters:
            assert "entity" in cluster
            assert "similar" in cluster

        # Validate subset relationship structure
        for subset in subsets:
            assert "subsetGroup" in subset
            assert "supersetGroup" in subset
            assert isinstance(subset["subsetGroup"], str)
            assert isinstance(subset["supersetGroup"], str)

    def test_entity_clustering_with_merging_enabled(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test entity clustering with merge_similar_entities enabled."""
        result = vector_store_with_data.find_entity_clusters(
            threshold=0.6,
            k=4,
            use_approx=False,
            use_subset_relations=True,
            merge_similar_entities=True,
        )

        # Should return a dictionary with all three keys
        assert isinstance(result, dict)
        assert "similar_entities" in result
        assert "subset_relationships" in result
        assert "merged_entities" in result

        similar_entities = result["similar_entities"]
        subset_relationships = result["subset_relationships"]
        merged_entities = result["merged_entities"]

        assert isinstance(similar_entities, list)
        assert isinstance(subset_relationships, list)
        assert isinstance(merged_entities, list)
        assert len(similar_entities) > 0

        # Validate merged entities structure
        for merged_cluster in merged_entities:
            assert "entity" in merged_cluster
            assert "merged_entities" in merged_cluster
            assert isinstance(merged_cluster["entity"], str)
            assert isinstance(merged_cluster["merged_entities"], list)

        # If subset relationships exist, merged entities should be fewer or equal
        if subset_relationships:
            assert len(merged_entities) <= len(similar_entities)

    def test_entity_clustering_merging_warning_case(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test warning when merge_similar_entities=True but
        use_subset_relations=False."""
        import warnings

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = vector_store_with_data.find_entity_clusters(
                threshold=0.7,
                k=3,
                use_approx=False,
                use_subset_relations=False,  # False
                merge_similar_entities=True,  # True - should trigger warning
            )

            # Should return basic clusters and issue warning
            assert isinstance(result, list)
            assert len(w) == 1
            expected_msg = (
                "merge_similar_entities=True requires use_subset_relations=True"
            )
            assert expected_msg in str(w[0].message)
            assert issubclass(w[0].category, UserWarning)

    def test_different_threshold_values(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test entity clustering with different similarity thresholds."""
        thresholds = [0.1, 0.5, 0.8]
        results = {}

        for threshold in thresholds:
            result = vector_store_with_data.find_entity_clusters(
                threshold=threshold, k=3, use_approx=False
            )
            results[threshold] = len(result)

        # Higher thresholds should generally produce fewer clusters
        assert all(isinstance(count, int) and count >= 0 for count in results.values())

    def test_different_distance_strategies(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test entity clustering with different distance strategies."""
        # Test Cosine similarity (default)
        cosine_result = vector_store_with_data.find_entity_clusters(
            threshold=0.7, k=3, use_approx=False
        )

        # Test Euclidean distance
        vector_store_with_data._distance_strategy = DistanceStrategy.EUCLIDEAN_DISTANCE
        euclidean_result = vector_store_with_data.find_entity_clusters(
            threshold=0.1,  # Lower threshold for Euclidean
            k=3,
            use_approx=False,
        )

        # Both should return valid results
        assert isinstance(cosine_result, list)
        assert isinstance(euclidean_result, list)
        assert len(cosine_result) > 0
        assert len(euclidean_result) > 0

    def test_empty_results_with_high_threshold(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test entity clustering with very high threshold returning no results."""
        # Test with very high threshold - should return empty results
        result = vector_store_with_data.find_entity_clusters(
            threshold=0.99999,  # Extremely high threshold
            k=3,
            use_approx=False,
            use_subset_relations=False,
        )
        assert result == []

        # Test with subset relations and high threshold
        result_with_subsets = vector_store_with_data.find_entity_clusters(
            threshold=0.99999,  # Extremely high threshold
            k=3,
            use_approx=False,
            use_subset_relations=True,
        )
        assert result_with_subsets == {
            "similar_entities": [],
            "subset_relationships": [],
        }

    def test_single_document_collection(self, db: StandardDatabase) -> None:
        """Test entity clustering with single document collection."""
        collection_name = "test_single_clustering"

        # Clean up any existing collection
        if db.has_collection(collection_name):
            db.delete_collection(collection_name)

        try:
            fake_embeddings = FakeEmbeddings(dimension=EMBEDDING_DIMENSION)

            # Create vector store with single document
            vector_store = ArangoVector.from_texts(
                texts=["Single document for testing"],
                embedding=fake_embeddings,
                database=db,
                collection_name=collection_name,
                ids=["doc1"],
            )

            # Test clustering - should return empty since no similar documents
            result = vector_store.find_entity_clusters(
                threshold=0.5, k=3, use_subset_relations=False
            )
            assert result == []

        finally:
            # Cleanup
            if db.has_collection(collection_name):
                db.delete_collection(collection_name)

    def test_error_handling_invalid_parameters(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test error handling with various parameter combinations."""
        # Very high threshold should return empty results, not error
        result = vector_store_with_data.find_entity_clusters(
            threshold=0.99, k=3, use_approx=False
        )
        assert isinstance(result, list)

        # k=0 should still work (return no similar entities)
        result = vector_store_with_data.find_entity_clusters(
            threshold=0.6, k=0, use_approx=False
        )
        assert isinstance(result, list)

        # Very large k should be handled gracefully
        result = vector_store_with_data.find_entity_clusters(
            threshold=0.1, k=1000, use_approx=False
        )
        assert isinstance(result, list)

    def test_unsupported_distance_strategy(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test error handling for unsupported distance strategies."""
        # Set invalid distance strategy
        vector_store_with_data._distance_strategy = "INVALID_STRATEGY"  # type: ignore

        with pytest.raises(ValueError, match="Unsupported metric"):
            vector_store_with_data.find_entity_clusters(
                threshold=0.5, k=3, use_approx=False
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
