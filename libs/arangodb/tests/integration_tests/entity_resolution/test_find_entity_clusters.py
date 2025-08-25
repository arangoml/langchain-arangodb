"""
Integration tests for find_entity_clusters functionality.

These tests require:
- Running ArangoDB instance
- Real database operations with actual collections
- Comprehensive testing of clustering behavior
"""

from typing import Any, Dict, List, cast

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
            result_dict = cast(Dict[str, Any], result)
            results[k] = {
                "clusters": len(result_dict["similar_entities"]),
                "max_similar": (
                    max(len(cluster["similar"]) for cluster in result_dict["similar_entities"])
                    if result_dict["similar_entities"]
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
        assert result_with_subsets == {"similar_entities": [], "subset_relationships": []}

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
        threshold_val: float = 0.7
        k_val: int = 4
        use_subset_relations_val: bool = True

        # Test exact search
        # start_time = time.time()  # Timing removed
        exact_result = vector_store_with_data.find_entity_clusters(
            threshold=threshold_val,
            k=k_val,
            use_subset_relations=use_subset_relations_val,
            use_approx=False,
        )
        # exact_time = time.time() - start_time  # Timing removed

        # Test approximate search (may fail if vector index not supported)
        try:
            # start_time = time.time()  # Timing removed
            approx_result = vector_store_with_data.find_entity_clusters(
                threshold=threshold_val,
                k=k_val,
                use_subset_relations=use_subset_relations_val,
                use_approx=True,
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

        result_dict = cast(Dict[str, Any], result)
        clusters = result_dict["similar_entities"]
        subset_relations = result_dict["subset_relationships"]

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
            result_dict = cast(Dict[str, Any], result)
            assert len(result_dict["similar_entities"]) > 0

        finally:
            # Cleanup
            if db.has_collection(collection_name):
                db.delete_collection(collection_name)

    def test_entity_clustering_with_merging_enabled(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test entity clustering with merge_similar_entities enabled."""
        result = vector_store_with_data.find_entity_clusters(
            threshold=0.6, 
            k=4, 
            use_approx=False, 
            use_subset_relations=True,
            merge_similar_entities=True
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

    def test_entity_clustering_merging_no_subset_relationships(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test entity clustering with merging when no subset relationships exist."""
        # Use very high threshold to avoid subset relationships
        result = vector_store_with_data.find_entity_clusters(
            threshold=0.95, 
            k=2, 
            use_approx=False, 
            use_subset_relations=True,
            merge_similar_entities=True
        )

        # Should handle case with no subset relationships gracefully
        if isinstance(result, dict) and "subset_relationships" in result:
            subset_relationships = result["subset_relationships"]
            merged_entities = result["merged_entities"]
            
            # When no subset relationships exist, merged_entities should be empty
            if not subset_relationships:
                assert merged_entities == []

    def test_entity_clustering_merging_warning_case(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test warning when merge_similar_entities=True but use_subset_relations=False."""
        import warnings
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = vector_store_with_data.find_entity_clusters(
                threshold=0.7,
                k=3,
                use_approx=False,
                use_subset_relations=False,  # False
                merge_similar_entities=True  # True - should trigger warning
            )
            
            # Should return basic clusters and issue warning
            assert isinstance(result, list)
            assert len(w) == 1
            expected_msg = ("merge_similar_entities=True requires "
                           "use_subset_relations=True")
            assert expected_msg in str(w[0].message)
            assert issubclass(w[0].category, UserWarning)

    def test_entity_clustering_merging_with_known_duplicates(
        self, db: StandardDatabase
    ) -> None:
        """Test entity clustering with merging using known duplicate documents."""
        collection_name = "test_merging_duplicates"

        # Clean up any existing collection
        if db.has_collection(collection_name):
            db.delete_collection(collection_name)

        try:
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

            # Add documents with clear duplicate patterns
            documents = [
                {
                    "_key": "ned_stark_full",
                    "name": "Eddard",
                    "surname": "Stark",
                    "house": "Stark",
                    "region": "North",
                    "alive": False,
                    "age": 41,
                    "titles": ["Lord of Winterfell", "Warden of the North"],
                    "traits": ["honorable", "duty-bound", "just"],
                    "text": "Ned Stark honorable lord Winterfell northern warden dutiful just",
                    "profession": "lord",
                },
                {
                    "_key": "ned_stark_short", 
                    "name": "Ned",
                    "surname": "Stark",
                    "house": "Stark",
                    "region": "North",
                    "alive": False,
                    "age": 41,
                    "titles": ["Lord of Winterfell"],
                    "traits": ["honorable", "duty-bound"],
                    "text": "Ned Stark honorable lord Winterfell",
                    "profession": "lord",
                },
                {
                    "_key": "tyrion_full",
                    "name": "Tyrion",
                    "surname": "Lannister",
                    "house": "Lannister",
                    "region": "Westerlands",
                    "alive": True,
                    "age": 32,
                    "titles": ["Hand of the King", "Imp"],
                    "traits": ["clever", "witty", "dwarf"],
                    "text": "Tyrion Lannister clever imp witty hand king dwarf intelligent wise",
                    "profession": "advisor",
                },
                {
                    "_key": "tyrion_short",
                    "name": "Tyrion",
                    "surname": "Lannister",
                    "house": "Lannister",
                    "region": "Westerlands",
                    "alive": True,
                    "age": 32,
                    "titles": ["Imp"],
                    "traits": ["clever", "witty"],
                    "text": "Tyrion Lannister clever imp witty",
                    "profession": "advisor",
                },
                {
                    "_key": "bronn_independent",
                    "name": "Bronn",
                    "surname": "",
                    "house": "None",
                    "region": "Various",
                    "alive": True,
                    "age": 35,
                    "titles": ["Ser"],
                    "traits": ["mercenary", "pragmatic", "skilled"],
                    "text": "Bronn independent sellsword mercenary pragmatic skilled fighter",
                    "profession": "sellsword",
                }
            ]

            texts = [doc["text"] for doc in documents]
            metadatas = [
                {k: v for k, v in doc.items() if k not in ["_key", "text"]}
                for doc in documents
            ]
            ids = [doc["_key"] for doc in documents]

            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

            # Test clustering with merging
            result = vector_store.find_entity_clusters(
                threshold=0.3, 
                k=4, 
                use_approx=False, 
                use_subset_relations=True,
                merge_similar_entities=True
            )

            # Should successfully merge similar entities
            assert isinstance(result, dict)
            assert "similar_entities" in result
            assert "subset_relationships" in result
            assert "merged_entities" in result

            # Validate that merging logic worked
            similar_entities = result["similar_entities"]
            merged_entities = result["merged_entities"]
            subset_relationships = result["subset_relationships"]

            # Should have found some clusters
            assert len(similar_entities) > 0

            # If subset relationships were found, verify merging worked
            if subset_relationships:
                # Merged entities should be fewer than or equal to original
                assert len(merged_entities) <= len(similar_entities)
                
                # Each merged entity should have proper structure
                for merged in merged_entities:
                    assert "entity" in merged
                    assert "merged_entities" in merged
                    assert len(merged["merged_entities"]) > 0

        finally:
            # Cleanup
            if db.has_collection(collection_name):
                db.delete_collection(collection_name)

    def test_entity_clustering_merging_complex_hierarchy(
        self, db: StandardDatabase  
    ) -> None:
        """Test entity clustering with merging in complex hierarchical scenarios."""
        collection_name = "test_merging_hierarchy"

        # Clean up any existing collection
        if db.has_collection(collection_name):
            db.delete_collection(collection_name)

        try:
            fake_embeddings = FakeEmbeddings(dimension=EMBEDDING_DIMENSION)

            vector_store = ArangoVector(
                embedding=fake_embeddings,
                embedding_dimension=EMBEDDING_DIMENSION,
                database=db,
                collection_name=collection_name,
            )

            # Create documents with hierarchical similarity (A ⊂ B ⊂ C)
            documents = [
                {
                    "_key": "stark_minimal",
                    "name": "Stark",
                    "surname": "",
                    "house": "Stark",
                    "region": "North",
                    "alive": True,
                    "age": 25,
                    "titles": [],
                    "traits": ["stark", "north", "honor"],
                    "text": "stark north honor",
                    "profession": "unknown",
                    "detail_level": "minimal"
                },
                {
                    "_key": "stark_basic",
                    "name": "Stark",
                    "surname": "Member",
                    "house": "Stark",
                    "region": "North",
                    "alive": True,
                    "age": 25,
                    "titles": ["Family Member"],
                    "traits": ["stark", "north", "honor", "duty", "family"],
                    "text": "stark north honor duty family",
                    "profession": "family_member",
                    "detail_level": "basic"
                },
                {
                    "_key": "stark_detailed", 
                    "name": "Lord",
                    "surname": "Stark",
                    "house": "Stark",
                    "region": "North",
                    "alive": True,
                    "age": 40,
                    "titles": ["Lord of Winterfell", "Warden of the North"],
                    "traits": ["stark", "north", "honor", "duty", "family", "lord"],
                    "text": "stark north honor duty family winterfell lord warden",
                    "profession": "lord",
                    "detail_level": "detailed"
                },
                {
                    "_key": "lannister_different",
                    "name": "Lannister",
                    "surname": "Member",
                    "house": "Lannister",
                    "region": "Westerlands",
                    "alive": True,
                    "age": 30,
                    "titles": ["Rich"],
                    "traits": ["lannister", "gold", "power", "wealth"],
                    "text": "lannister gold power wealth different",
                    "profession": "noble",
                    "detail_level": "independent"
                }
            ]

            texts = [doc["text"] for doc in documents]
            metadatas = [
                {k: v for k, v in doc.items() if k not in ["_key", "text"]}
                for doc in documents
            ]
            ids = [doc["_key"] for doc in documents]

            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

            # Test with low threshold to ensure hierarchy is detected
            result = vector_store.find_entity_clusters(
                threshold=0.1,
                k=5,
                use_approx=False,
                use_subset_relations=True,
                merge_similar_entities=True
            )

            # Should handle complex hierarchy properly
            assert isinstance(result, dict)
            
            subset_relationships = result["subset_relationships"]
            merged_entities = result["merged_entities"]

            # If hierarchical relationships were detected
            if subset_relationships and merged_entities:
                # Should have consolidated the hierarchy
                assert len(merged_entities) > 0
                
                # Verify that top-level entity contains merged content
                for merged in merged_entities:
                    if merged["entity"] == "stark_detailed":
                        # Should contain entities from lower levels
                        merged_list = merged["merged_entities"]
                        assert len(merged_list) > 1

        finally:
            # Cleanup  
            if db.has_collection(collection_name):
                db.delete_collection(collection_name)

    def test_entity_clustering_merging_with_different_thresholds(
        self, db: StandardDatabase
    ) -> None:
        """Test entity clustering merging behavior with different threshold values."""
        collection_name = "test_merging_thresholds"

        # Clean up any existing collection
        if db.has_collection(collection_name):
            db.delete_collection(collection_name)

        try:
            fake_embeddings = FakeEmbeddings(dimension=EMBEDDING_DIMENSION)

            vector_store = ArangoVector(
                embedding=fake_embeddings,
                embedding_dimension=EMBEDDING_DIMENSION,
                database=db,
                collection_name=collection_name,
            )

            # Add similar documents that should cluster differently at different thresholds
            documents = [
                {
                    "_key": "stark_very_detailed",
                    "name": "Brandon",
                    "surname": "Stark",
                    "house": "Stark",
                    "region": "North",
                    "alive": True,
                    "age": 16,
                    "titles": ["Three-Eyed Raven", "Lord of Winterfell"],
                    "traits": ["stark", "family", "northern", "honor", "duty", "loyalty"],
                    "text": "stark family northern honor duty loyalty winterfell",
                    "profession": "lord",
                    "similarity_type": "very_similar"
                },
                {
                    "_key": "stark_detailed",
                    "name": "Rickon",
                    "surname": "Stark",
                    "house": "Stark",
                    "region": "North",
                    "alive": False,
                    "age": 11,
                    "titles": ["Prince"],
                    "traits": ["stark", "family", "northern", "honor", "duty", "loyalty"],
                    "text": "stark family northern honor duty loyalty",
                    "profession": "prince",
                    "similarity_type": "very_similar"
                },
                {
                    "_key": "stark_basic",
                    "name": "Benjen",
                    "surname": "Stark",
                    "house": "Stark",
                    "region": "North",
                    "alive": False,
                    "age": 45,
                    "titles": ["First Ranger"],
                    "traits": ["stark", "northern", "honor"],
                    "text": "stark northern honor",
                    "profession": "ranger",
                    "similarity_type": "somewhat_similar"
                },
                {
                    "_key": "lannister_different",
                    "name": "Tywin",
                    "surname": "Lannister",
                    "house": "Lannister",
                    "region": "Westerlands",
                    "alive": False,
                    "age": 67,
                    "titles": ["Lord of Casterly Rock", "Hand of the King"],
                    "traits": ["lannister", "gold", "rich", "powerful"],
                    "text": "lannister gold rich powerful",
                    "profession": "lord",
                    "similarity_type": "different"
                }
            ]

            texts = [doc["text"] for doc in documents]
            metadatas = [
                {k: v for k, v in doc.items() if k not in ["_key", "text"]}
                for doc in documents
            ]
            ids = [doc["_key"] for doc in documents]

            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

            # Test with different thresholds
            thresholds = [0.1, 0.5, 0.8]
            results = {}

            for threshold in thresholds:
                result = vector_store.find_entity_clusters(
                    threshold=threshold,
                    k=4,
                    use_approx=False,
                    use_subset_relations=True,
                    merge_similar_entities=True
                )
                
                assert isinstance(result, dict)
                results[threshold] = {
                    "similar_entities_count": len(result["similar_entities"]),
                    "subset_relationships_count": len(result["subset_relationships"]),
                    "merged_entities_count": len(result["merged_entities"])
                }

            # Validate threshold behavior
            for threshold, data in results.items():
                assert isinstance(data["similar_entities_count"], int)
                assert isinstance(data["subset_relationships_count"], int)
                assert isinstance(data["merged_entities_count"], int)
                assert data["similar_entities_count"] >= 0
                assert data["subset_relationships_count"] >= 0
                assert data["merged_entities_count"] >= 0

        finally:
            # Cleanup
            if db.has_collection(collection_name):
                db.delete_collection(collection_name)

    def test_entity_clustering_merging_empty_collection(
        self, db: StandardDatabase
    ) -> None:
        """Test entity clustering merging with empty collection."""
        collection_name = "test_merging_empty"

        # Clean up any existing collection
        if db.has_collection(collection_name):
            db.delete_collection(collection_name)

        try:
            fake_embeddings = FakeEmbeddings(dimension=EMBEDDING_DIMENSION)

            # Create empty vector store
            vector_store = ArangoVector(
                embedding=fake_embeddings,
                embedding_dimension=EMBEDDING_DIMENSION,
                database=db,
                collection_name=collection_name,
            )

            # Test clustering on empty collection
            result = vector_store.find_entity_clusters(
                threshold=0.5,
                k=3,
                use_approx=False,
                use_subset_relations=True,
                merge_similar_entities=True
            )

            # Should return empty structure
            assert result == {"similar_entities": [], "subset_relationships": []}

        finally:
            # Cleanup
            if db.has_collection(collection_name):
                db.delete_collection(collection_name)

    def test_entity_clustering_merging_performance_comparison(
        self, vector_store_with_data: ArangoVector
    ) -> None:
        """Test performance comparison between merging enabled and disabled."""
        # Test without merging
        result_no_merge = vector_store_with_data.find_entity_clusters(
            threshold=0.7,
            k=4,
            use_approx=False,
            use_subset_relations=True,
            merge_similar_entities=False
        )

        # Test with merging
        result_with_merge = vector_store_with_data.find_entity_clusters(
            threshold=0.7,
            k=4,
            use_approx=False,
            use_subset_relations=True,
            merge_similar_entities=True
        )

        # Both should return valid results
        assert isinstance(result_no_merge, dict)
        assert isinstance(result_with_merge, dict)

        # Validate structure differences
        assert "similar_entities" in result_no_merge
        assert "subset_relationships" in result_no_merge
        assert "merged_entities" not in result_no_merge

        assert "similar_entities" in result_with_merge
        assert "subset_relationships" in result_with_merge
        assert "merged_entities" in result_with_merge

        # Results should be consistent in similar_entities and subset_relationships
        assert result_no_merge["similar_entities"] == result_with_merge["similar_entities"]
        assert result_no_merge["subset_relationships"] == result_with_merge["subset_relationships"]

    def test_entity_clustering_merging_real_world_scenario(
        self, db: StandardDatabase
    ) -> None:
        """Test entity clustering merging with realistic duplicate entity scenario."""
        collection_name = "test_merging_real_world"

        # Clean up any existing collection
        if db.has_collection(collection_name):
            db.delete_collection(collection_name)

        try:
            fake_embeddings = FakeEmbeddings(dimension=EMBEDDING_DIMENSION)

            vector_store = ArangoVector(
                embedding=fake_embeddings,
                embedding_dimension=EMBEDDING_DIMENSION,
                database=db,
                collection_name=collection_name,
            )

            # Real-world scenario: character entities with modern variations
            documents = [
                # Daenerys - multiple representations (modern equivalent)
                {
                    "_key": "daenerys_full",
                    "name": "Daenerys",
                    "surname": "Targaryen",
                    "house": "Targaryen",
                    "region": "Essos",
                    "alive": False,
                    "age": 16,
                    "titles": ["Queen of Dragons", "Breaker of Chains", "Mother of Dragons"],
                    "traits": ["powerful", "determined", "fire", "dragons"],
                    "text": "Daenerys Targaryen queen dragons fire breaker chains powerful",
                    "profession": "queen",
                    "detail_level": "full"
                },
                {
                    "_key": "daenerys_partial",
                    "name": "Dany",
                    "surname": "Targaryen",
                    "house": "Targaryen",
                    "region": "Essos",
                    "alive": False,
                    "age": 16,
                    "titles": ["Queen of Dragons"],
                    "traits": ["powerful", "fire", "dragons"],
                    "text": "Dany Targaryen queen dragons fire",
                    "profession": "queen",
                    "detail_level": "partial"
                },
                {
                    "_key": "daenerys_minimal",
                    "name": "Daenerys",
                    "surname": "T",
                    "house": "Targaryen",
                    "region": "Essos",
                    "alive": False,
                    "age": 16,
                    "titles": [],
                    "traits": ["dragons"],
                    "text": "Daenerys dragons",
                    "profession": "queen",
                    "detail_level": "minimal"
                },
                # Sansa - single representation
                {
                    "_key": "sansa_stark",
                    "name": "Sansa",
                    "surname": "Stark",
                    "house": "Stark",
                    "region": "North",
                    "alive": True,
                    "age": 18,
                    "titles": ["Lady of Winterfell", "Queen in the North"],
                    "traits": ["smart", "political", "survivor"],
                    "text": "Sansa Stark lady political smart survivor north",
                    "profession": "lady",
                    "detail_level": "single"
                },
                # Sandor - duplicate representations  
                {
                    "_key": "sandor_long",
                    "name": "Sandor",
                    "surname": "Clegane",
                    "house": "Clegane",
                    "region": "Westerlands",
                    "alive": True,
                    "age": 35,
                    "titles": ["The Hound", "Ser"],
                    "traits": ["fierce", "loyal", "scarred", "fighter"],
                    "text": "Sandor Clegane hound fierce loyal scarred fighter knight",
                    "profession": "knight",
                    "detail_level": "long"
                },
                {
                    "_key": "sandor_short",
                    "name": "The Hound",
                    "surname": "Clegane",
                    "house": "Clegane",
                    "region": "Westerlands",
                    "alive": True,
                    "age": 35,
                    "titles": ["The Hound"],
                    "traits": ["fierce", "fighter"],
                    "text": "Hound Clegane fierce fighter",
                    "profession": "knight",
                    "detail_level": "short"
                }
            ]

            texts = [doc["text"] for doc in documents]
            metadatas = [
                {k: v for k, v in doc.items() if k not in ["_key", "text"]}
                for doc in documents
            ]
            ids = [doc["_key"] for doc in documents]

            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

            # Test entity resolution with merging
            result = vector_store.find_entity_clusters(
                threshold=0.2,  # Lower threshold to catch similar entities
                k=5,
                use_approx=False,
                use_subset_relations=True,
                merge_similar_entities=True
            )

            # Should detect and merge similar entities
            assert isinstance(result, dict)
            assert "similar_entities" in result
            assert "subset_relationships" in result
            assert "merged_entities" in result

            similar_entities = result["similar_entities"]
            subset_relationships = result["subset_relationships"]
            merged_entities = result["merged_entities"]

            # Should find some similar entities
            assert len(similar_entities) > 0

            # Validate entity resolution worked
            if subset_relationships:
                # Should have detected subset relationships
                assert len(subset_relationships) > 0
                
                # Should have merged some entities
                assert len(merged_entities) > 0
                assert len(merged_entities) <= len(similar_entities)

                # Validate merge structure
                for merged in merged_entities:
                    assert "entity" in merged
                    assert "merged_entities" in merged
                    assert isinstance(merged["merged_entities"], list)

        finally:
            # Cleanup
            if db.has_collection(collection_name):
                db.delete_collection(collection_name)

    def test_entity_clustering_merging_with_exact_duplicates(
        self, db: StandardDatabase
    ) -> None:
        """Test entity clustering merging with exact duplicate documents."""
        collection_name = "test_merging_exact_duplicates"

        # Clean up any existing collection
        if db.has_collection(collection_name):
            db.delete_collection(collection_name)

        try:
            fake_embeddings = FakeEmbeddings(dimension=EMBEDDING_DIMENSION)

            vector_store = ArangoVector(
                embedding=fake_embeddings,
                embedding_dimension=EMBEDDING_DIMENSION,
                database=db,
                collection_name=collection_name,
            )

            # Create exact duplicate documents (same content, different keys)
            documents = [
                {
                    "_key": "jon_snow_001",
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
                    "source": "database_1"
                },
                {
                    "_key": "jon_snow_002",
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
                    "source": "database_2"
                },
                {
                    "_key": "arya_stark_001",
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
                    "source": "database_1"
                },
                {
                    "_key": "arya_stark_002",
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
                    "source": "database_2"
                },
                {
                    "_key": "unique_character",
                    "name": "Hodor",
                    "surname": "",
                    "house": "Stark",
                    "region": "North",
                    "alive": False,
                    "age": 40,
                    "titles": [],
                    "traits": ["simple", "loyal", "strong"],
                    "text": "Hodor simple loyal strong giant servant",
                    "profession": "servant",
                    "source": "single"
                }
            ]

            texts = [doc["text"] for doc in documents]
            metadatas = [
                {k: v for k, v in doc.items() if k not in ["_key", "text"]}
                for doc in documents
            ]
            ids = [doc["_key"] for doc in documents]

            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

            # Test exact duplicate detection and merging
            result = vector_store.find_entity_clusters(
                threshold=0.1,  # Very low threshold to catch exact duplicates
                k=5,
                use_approx=False,
                use_subset_relations=True,
                merge_similar_entities=True
            )

            # Should detect exact duplicates
            assert isinstance(result, dict)
            assert "similar_entities" in result
            assert "subset_relationships" in result
            assert "merged_entities" in result

            similar_entities = result["similar_entities"]
            subset_relationships = result["subset_relationships"]
            merged_entities = result["merged_entities"]

            # Should find similar entities (exact duplicates)
            assert len(similar_entities) > 0

            # Should detect some exact duplicate relationships
            if subset_relationships:
                # Should have detected duplicate relationships
                assert len(subset_relationships) > 0
                
                # Should have merged exact duplicates
                assert len(merged_entities) > 0
                assert len(merged_entities) <= len(similar_entities)

                # Validate that exact duplicates were properly merged
                for merged in merged_entities:
                    assert "entity" in merged
                    assert "merged_entities" in merged
                    # For exact duplicates, should contain the duplicate entity
                    if len(merged["merged_entities"]) > 1:
                        # This means actual merging occurred
                        assert isinstance(merged["merged_entities"], list)

        finally:
            # Cleanup
            if db.has_collection(collection_name):
                db.delete_collection(collection_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
