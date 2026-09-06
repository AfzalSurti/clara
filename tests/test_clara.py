import random
import unittest

from clara import CLARA
from clara.metrics import average_cost, silhouette_score


def make_blobs(rng, centers, per, spread=1.0):
    pts, truth = [], []
    for ci, c in enumerate(centers):
        for _ in range(per):
            pts.append([coord + rng.gauss(0, spread) for coord in c])
            truth.append(ci)
    return pts, truth


def contingency_pure(truth, pred, k):
    """True if every predicted cluster maps to exactly one ground-truth class."""
    groups = {}
    for t, p in zip(truth, pred):
        groups.setdefault(p, set()).add(t)
    return len(groups) == k and all(len(v) == 1 for v in groups.values())


class TestCLARA(unittest.TestCase):
    def test_recovers_clusters_on_large_input(self):
        rng = random.Random(42)
        centers = [(0.0, 0.0), (25.0, 0.0), (0.0, 25.0), (25.0, 25.0)]
        pts, truth = make_blobs(rng, centers, per=300, spread=1.5)
        model = CLARA(n_clusters=4, n_samples=5, random_state=0).fit(pts)

        self.assertEqual(len(model.labels_), len(pts))
        self.assertEqual(set(model.labels_), {0, 1, 2, 3})
        self.assertTrue(contingency_pure(truth, model.labels_, 4))
        self.assertGreater(silhouette_score(pts, model.labels_), 0.8)

    def test_is_reproducible_with_seed(self):
        rng = random.Random(7)
        pts, _ = make_blobs(rng, [(0, 0), (20, 20), (0, 20)], per=120, spread=2.0)
        a = CLARA(n_clusters=3, random_state=99).fit(pts)
        b = CLARA(n_clusters=3, random_state=99).fit(pts)
        self.assertEqual(a.medoid_indices_, b.medoid_indices_)
        self.assertEqual(a.labels_, b.labels_)
        self.assertAlmostEqual(a.cost_, b.cost_)

    def test_more_samples_do_not_worsen_cost(self):
        rng = random.Random(3)
        pts, _ = make_blobs(rng, [(0, 0), (15, 0), (0, 15)], per=150, spread=2.5)
        few = CLARA(n_clusters=3, n_samples=1, random_state=1).fit(pts)
        many = CLARA(n_clusters=3, n_samples=8, random_state=1).fit(pts)
        # the first sample is identical (same seed), later ones only replace the
        # best if they beat it, so more samples can only help or tie
        self.assertLessEqual(many.cost_ - 1e-9, few.cost_)

    def test_cost_attributes_consistent(self):
        rng = random.Random(5)
        pts, _ = make_blobs(rng, [(0, 0), (10, 10)], per=80, spread=1.0)
        model = CLARA(n_clusters=2, random_state=2).fit(pts)
        recomputed = average_cost(pts, model.labels_, model.medoids_)
        self.assertAlmostEqual(recomputed, model.cost_, places=9)
        self.assertAlmostEqual(model.total_cost_, model.cost_ * len(pts))
        for gi in model.medoid_indices_:
            self.assertIn(list(pts[gi]), [list(m) for m in model.medoids_])

    def test_medoids_are_real_points(self):
        rng = random.Random(8)
        pts, _ = make_blobs(rng, [(0, 0), (9, 9), (0, 9)], per=60, spread=1.0)
        model = CLARA(n_clusters=3, random_state=4).fit(pts)
        for m in model.medoids_:
            self.assertIn(m, [list(p) for p in pts])

    def test_predict_matches_labels_on_training_data(self):
        rng = random.Random(6)
        pts, _ = make_blobs(rng, [(0, 0), (12, 0), (0, 12)], per=70, spread=1.0)
        model = CLARA(n_clusters=3, random_state=1).fit(pts)
        self.assertEqual(model.predict(pts), model.labels_)

    def test_manhattan_metric(self):
        rng = random.Random(10)
        pts, truth = make_blobs(rng, [(0, 0), (18, 0), (0, 18)], per=100, spread=1.0)
        model = CLARA(n_clusters=3, metric="manhattan", random_state=0).fit(pts)
        self.assertTrue(contingency_pure(truth, model.labels_, 3))

    def test_small_n_falls_back_gracefully(self):
        pts = [[0.0], [0.2], [10.0], [10.1], [10.2]]
        model = CLARA(n_clusters=2, sample_size=None, random_state=0).fit(pts)
        self.assertEqual(model.labels_[0], model.labels_[1])
        self.assertEqual(model.labels_[2], model.labels_[3])
        self.assertNotEqual(model.labels_[0], model.labels_[2])

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            CLARA(n_clusters=0)
        with self.assertRaises(ValueError):
            CLARA(n_clusters=3, sample_size=3)
        with self.assertRaises(ValueError):
            CLARA(n_clusters=2, n_samples=0)
        with self.assertRaises(ValueError):
            CLARA(n_clusters=5).fit([[0.0], [1.0]])


if __name__ == "__main__":
    unittest.main()
