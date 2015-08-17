import numpy as np

from datasets import Joints
from pairwise_relations import from_dataset


def generate_fake_locations(num, means, stddev=5):
    """Generate a matrix with four rows (one for each "point") and three
    columns (x-coord, y-coord and visibility). Means is a 3x2 matrix giving
    mean locations for each point."""
    per_joint = []
    for joint_mean in means:
        locations = np.random.multivariate_normal(
            joint_mean, stddev * np.eye(2), num
        )
        with_visibility = np.append(locations, np.ones((num, 1)), axis=1)
        per_joint.append(with_visibility)
    warped_array = np.array(per_joint)
    # Now we need to swap the first and second dimensions
    return warped_array.transpose((1, 0, 2))


def test_clustering():
    """Test learning of clusters for joint types."""
    first_means = np.asarray([
        (10, 70),
        (58, 94),
        (66, 58),
        (95, 62)
    ])
    second_means = np.asarray([
        (88, 12),
        (56, 15),
        (25, 21),
        (24, 89)
    ])
    fake_locations = np.concatenate([
        generate_fake_locations(100, first_means),
        generate_fake_locations(100, second_means),
    ], axis=0)
    np.random.shuffle(fake_locations)
    fake_pairs = [
        (0, 1),
        (1, 2),
        (2, 3)
    ]
    fake_joints = Joints(fake_locations, fake_pairs)
    # Make two clusters for each relationship type. Yes, passing in zeros as
    # your scale is stupid, and poor testing practice.
    centers = from_dataset(fake_joints, 2, np.zeros(len(fake_locations)), 1)

    assert centers.ndim == 3
    # Three joints, two clusters per joint, two coordinates (i.e. x, y) per
    # cluster
    assert centers.shape == (3, 2, 2)

    for idx, pair in enumerate(fake_pairs):
        first_idx, second_idx = pair
        first_mean = first_means[second_idx] - first_means[first_idx]
        second_mean = second_means[second_idx] - second_means[first_idx]
        found_means = centers[idx]
        first_dists = np.linalg.norm(found_means - first_mean, axis=1)
        second_dists = np.linalg.norm(found_means - second_mean, axis=1)

        # Make sure that each of our specified means are within Euclidean
        # distance 1 of at least one found cluster
        first_within = first_dists < 1
        assert first_within.any()
        second_within = second_dists < 1
        assert second_within.any()
