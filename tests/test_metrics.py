import math
import random
import unittest

from clara import metrics


def three_blobs(rng, per=15, spread=1.0):
    centers = [(0.0, 0.0), (30.0, 0.0), (0.0, 30.0)]
    pts, truth = [], []
    for ci, (cx, cy) in enumerate(centers):
        for _ in range(per):
            pts.append([cx + rng.gauss(0, spread), cy + rng.gauss(0, spread)])
            truth.append(ci)
    return pts, truth


class TestCost(unittest.TestCase):
    def test_total_and_average_cost(self):
        X = [[0.0], [0.0], [10.0], [10.0]]
        medoids = [[0.0], [10.0]]
        labels = [0, 0, 1, 1]
        self.assertAlmostEqual(metrics.total_cost(X, labels, medoids), 0.0)
        X2 = [[1.0], [-1.0], [12.0], [8.0]]
        self.assertAlmostEqual(metrics.total_cost(X2, labels, medoids), 1 + 1 + 2 + 2)
        self.assertAlmostEqual(metrics.average_cost(X2, labels, medoids), 6.0 / 4)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            metrics.total_cost([[0.0]], [0, 1], [[0.0]])


class TestSilhouette(unittest.TestCase):
    def test_well_separated_is_near_one(self):
        rng = random.Random(1)
        pts, truth = three_blobs(rng, per=25, spread=0.5)
        score = metrics.silhouette_score(pts, truth)
        self.assertGreater(score, 0.9)

    def test_wrong_labels_score_lower_than_true(self):
        rng = random.Random(2)
        pts, truth = three_blobs(rng, per=20, spread=1.0)
        good = metrics.silhouette_score(pts, truth)
        shuffled = truth[:]
        rng.shuffle(shuffled)
        bad = metrics.silhouette_score(pts, shuffled)
        self.assertGreater(good, bad)

    def test_single_cluster_scores_zero(self):
        pts = [[0.0], [1.0], [2.0], [3.0]]
        self.assertAlmostEqual(metrics.silhouette_score(pts, [0, 0, 0, 0]), 0.0)

    def test_singleton_cluster_point_is_zero(self):
        pts = [[0.0], [0.1], [100.0]]
        samples = metrics.silhouette_samples(pts, [0, 0, 1])
        self.assertAlmostEqual(samples[2], 0.0)

    def test_samples_in_range(self):
        rng = random.Random(3)
        pts, truth = three_blobs(rng, per=15, spread=2.0)
        for s in metrics.silhouette_samples(pts, truth):
            self.assertGreaterEqual(s, -1.0 - 1e-9)
            self.assertLessEqual(s, 1.0 + 1e-9)

    def test_matches_hand_computation_tiny(self):
        # two points per cluster, easy to do by hand
        pts = [[0.0], [2.0], [10.0], [12.0]]
        labels = [0, 0, 1, 1]
        s = metrics.silhouette_samples(pts, labels)
        # point 0: a = 2, b = mean(10, 12) = 11 -> (11-2)/11
        self.assertAlmostEqual(s[0], (11.0 - 2.0) / 11.0)
        # point 2: a = 2, b = mean(10, 8) = 9 -> (9-2)/9
        self.assertAlmostEqual(s[2], (9.0 - 2.0) / 9.0)


if __name__ == "__main__":
    unittest.main()
