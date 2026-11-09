"""Tests for selecting the global bank from skill embeddings."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from self_derived_skills.skill_bank import select_central_skills  # noqa: E402


def make_cluster(axis, size):
    """Build normalized vectors around one axis, where the first vector is the most central."""
    vectors = np.zeros((size, 4))
    vectors[:, axis] = 1.0
    other = (axis + 1) % 4
    for index in range(1, size):
        side = 1 if index % 2 else -1
        vectors[index, other] = side * 0.1 * ((index + 1) // 2)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_select_central_skills_takes_the_center_of_the_largest_clusters():
    """Clusters are ranked by size, and each contributes its most central member."""
    embeddings = np.vstack([make_cluster(0, 3), make_cluster(1, 5), make_cluster(2, 2)])
    embeddings = np.vstack([embeddings, make_cluster(3, 1)])
    assert select_central_skills(embeddings, bank_size=2, seed=230) == [(3, 5), (0, 3)]


def test_select_central_skills_needs_enough_distinct_skills():
    """Too few distinct skills is an error, not a smaller bank."""
    embeddings = np.vstack([make_cluster(0, 1)] * 5)
    with pytest.raises(ValueError, match="distinct skills"):
        select_central_skills(embeddings, bank_size=2, seed=230)
