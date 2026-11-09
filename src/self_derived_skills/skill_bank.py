"""Choose the skills in the global bank by clustering skill embeddings."""

import numpy as np
from sklearn.cluster import KMeans


def select_central_skills(
    embeddings: np.ndarray, bank_size: int, seed: int
) -> list[tuple[int, int]]:
    """Return (skill index, cluster size) for the central skill of each of the largest clusters.

    Embeddings must be L2-normalized, with shape (num_skills, dim). Clustering into twice the bank
    size is what lets the bank favor methods that recur across problems.
    """
    num_clusters = 2 * bank_size
    distinct = len(np.unique(embeddings, axis=0))
    if distinct < num_clusters:
        raise ValueError(f"Need {num_clusters} distinct skills to cluster, but found {distinct}")

    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed).fit(embeddings)
    sizes = np.bincount(kmeans.labels_, minlength=num_clusters)
    # A stable sort breaks ties by cluster index, which keeps the selection reproducible.
    largest = np.argsort(-sizes, kind="stable")[:bank_size]

    chosen = []
    for cluster in largest:
        members = np.flatnonzero(kmeans.labels_ == cluster)
        # All members share a centroid, so the dot product orders them the same way as cosine.
        similarity = embeddings[members] @ kmeans.cluster_centers_[cluster]
        chosen.append((int(members[np.argmax(similarity)]), int(sizes[cluster])))
    return chosen
