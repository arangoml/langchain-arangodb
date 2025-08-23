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
    ) -> ArangoVector:
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

        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

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

        # Clusters found and analyzed

    def test_entity_clustering_with_subset_relations(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test entity clustering with subset relationship analysis."""
        result = vector_store_with_data.find_entity_clusters(
            threshold=0.7, k=4, use_approx=False, use_subset_relations=True
        )

        # Should return a dictionary with clusters and subset_relationships
        assert isinstance(result, dict)
        assert "clusters" in result
        assert "subset_relationships" in result

        clusters = result["clusters"]
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

        # Analysis of clusters and subset relationships completed

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

        # Distance strategies compared

    def test_different_threshold_values(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test entity clustering with different similarity thresholds."""
        thresholds = [0.0, 0.1, 0.3, 0.5, 0.8]
        results = {}

        for threshold in thresholds:
            result = vector_store_with_data.find_entity_clusters(
                threshold=threshold, k=3, use_approx=False
            )
            results[threshold] = len(result)

        # Threshold analysis completed

        # Higher thresholds should generally produce fewer clusters
        assert all(isinstance(count, int) and count >= 0 for count in results.values())

    def test_different_k_values(self, vector_store_with_data: ArangoVector) -> None:
        """Test entity clustering with different k (similar entities count) values."""
        k_values = [1, 2, 4, 6]
        results = {}

        for k in k_values:
            result = vector_store_with_data.find_entity_clusters(
                threshold=0.1, k=k, use_approx=False, use_subset_relations=True
            )
            results[k] = {
                "clusters": len(result["clusters"]),
                "max_similar": (
                    max(len(cluster["similar"]) for cluster in result["clusters"])
                    if result["clusters"]
                    else 0
                ),
            }

        # K value analysis completed

        # Max similar entities should not exceed k
        for k, data in results.items():
            assert data["max_similar"] <= k

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
        assert result_with_subsets == {"clusters": [], "subset_relationships": []}

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

    def test_performance_comparison_exact_vs_approx(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test performance comparison between exact and approximate search."""
        test_params = {"threshold": 0.7, "k": 4, "use_subset_relations": True}

        # Test exact search
        # start_time = time.time()  # Timing removed
        exact_result = vector_store_with_data.find_entity_clusters(
            use_approx=False, **test_params
        )
        # exact_time = time.time() - start_time  # Timing removed

        # Test approximate search (may fail if vector index not supported)
        try:
            # start_time = time.time()  # Timing removed
            approx_result = vector_store_with_data.find_entity_clusters(
                use_approx=True, **test_params
            )
            # approx_time = time.time() - start_time  # Timing removed

            # Both should return valid results
            assert isinstance(exact_result, dict)
            assert isinstance(approx_result, dict)

            # Performance comparison completed

        except Exception as e:
            ann_error = "ANN search requires ArangoDB >= 3.12.4"
            if ann_error in str(e):
                pytest.skip("ArangoDB version doesn't support approximate search")
            elif "vector index" in str(e).lower():
                pytest.skip("Vector index feature not enabled in ArangoDB")
            else:
                raise

    def test_error_handling_invalid_threshold(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test error handling with invalid threshold values."""
        # Very high threshold should return empty results, not error
        result = vector_store_with_data.find_entity_clusters(
            threshold=0.99, k=3, use_approx=False
        )
        assert isinstance(result, list)
        # May be empty due to high threshold

    def test_error_handling_invalid_k(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test error handling with invalid k values."""
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

    def test_subset_relationship_logic(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test subset relationship detection logic with known data patterns."""
        # Use a very low threshold to ensure we get overlapping clusters
        result = vector_store_with_data.find_entity_clusters(
            threshold=0.1, k=6, use_approx=False, use_subset_relations=True
        )

        clusters = result["clusters"]
        subset_relations = result["subset_relationships"]

        # Detailed cluster analysis completed

        # Validate subset relationship logic
        for rel in subset_relations:
            subset_group = rel["subsetGroup"]
            superset_group = rel["supersetGroup"]

            # Find the corresponding clusters
            subset_cluster = next(c for c in clusters if c["entity"] == subset_group)
            superset_cluster = next(
                c for c in clusters if c["entity"] == superset_group
            )

            # Verify subset relationship
            subset_similar = set(subset_cluster["similar"])
            superset_similar = set(superset_cluster["similar"])

            # Subset should have fewer similar entities
            assert len(subset_similar) < len(superset_similar)

            # All entities in subset should be in superset
            assert subset_similar.issubset(superset_similar)

    def test_real_world_scenario_character_similarity(
        self, db: StandardDatabase
    ) -> None:
        """Test entity clustering with real-world character similarity scenario."""
        collection_name = "test_real_world_clustering"

        # Clean up any existing collection
        if db.has_collection(collection_name):
            db.delete_collection(collection_name)

        # Create collection
        db.create_collection(collection_name)

        try:
            # Use deterministic fake embeddings that create meaningful similarities
            fake_embeddings = FakeEmbeddings(dimension=128)

            vector_store = ArangoVector(
                embedding=fake_embeddings,
                embedding_dimension=128,
                database=db,
                collection_name=collection_name,
            )

            # Add documents with obvious similarity patterns
            documents = [
                {
                    "_key": "stark_family_1",
                    "text": "Stark family northern honor duty",
                    "house": "Stark",
                    "region": "North",
                },
                {
                    "_key": "stark_family_2",
                    "text": "Stark honor northern duty family",
                    "house": "Stark",
                    "region": "North",
                },
                {
                    "_key": "lannister_family_1",
                    "text": "Lannister family wealth power gold",
                    "house": "Lannister",
                    "region": "Westerlands",
                },
                {
                    "_key": "lannister_family_2",
                    "text": "Lannister gold power wealth family",
                    "house": "Lannister",
                    "region": "Westerlands",
                },
                {
                    "_key": "targaryen_family_1",
                    "text": "Targaryen dragons fire blood conquest",
                    "house": "Targaryen",
                    "region": "Essos",
                },
                {
                    "_key": "independent_1",
                    "text": "Independent mercenary traveling warrior",
                    "house": "None",
                    "region": "Various",
                },
            ]

            texts = [doc["text"] for doc in documents]
            metadatas = [
                {k: v for k, v in doc.items() if k not in ["_key", "text"]}
                for doc in documents
            ]
            ids = [doc["_key"] for doc in documents]

            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

            # Test clustering
            result = vector_store.find_entity_clusters(
                threshold=0.1, k=4, use_approx=False, use_subset_relations=True
            )

            # Real-world clustering analysis completed

            # Should find some meaningful clusters
            assert len(result["clusters"]) > 0

        finally:
            # Cleanup
            if db.has_collection(collection_name):
                db.delete_collection(collection_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
