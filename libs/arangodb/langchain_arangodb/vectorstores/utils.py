from enum import Enum


class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies for calculating distances
    between vectors.

    Note that **use_approx** is not supported for the following distance strategies:
    - JACCARD
    - MAX_INNER_PRODUCT
    - DOT_PRODUCT
    """

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    DOT_PRODUCT = "DOT_PRODUCT"
    JACCARD = "JACCARD"
    COSINE = "COSINE"
