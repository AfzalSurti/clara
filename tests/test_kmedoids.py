import random
import unittest

from clara.kmedoids import KMedoids, _assign, _build_init


def three_blobs(rng, per=15):
    centers = [(0.0, 0.0), (20.0, 0.0), (0.0, 20.0)]
    pts = []
    for cx, cy in centers:
        for _ in range(per):
            pts.append([cx + rng.gauss(0, 1.0), cy + rng.gauss(0, 1.0)])
    return pts, centers


class TestHelpers(unittest.TestCase):
    def test_assign_labels_and_cost(self):
        D = [
            [0.0, 1.0, 9.0],
            [1.0, 0.0, 8.0],
            [9.0, 8.0, 0.0],
        ]
        labels, cost = _assign(D, [0, 2])
        self.assertEqual(labels, [0, 0, 1])
        self.assertAlmostEqual(cost, 0.0 + 1.0 + 0.0)

    def test_build_init_picks_requested_count_distinct(self):
        rng = random.Random(1)
        pts, _ = three_blobs(rng)
        from clara.distance import euclidean, pairwise_matrix

        D = pairwise_matrix(pts, euclidean)
        medoids = _build_init(D, 3)
        self.assertEqual(len(medoids), 3)
        self.assertEqual(len(set(medoids)), 3)


class TestKMedoids(unittest.TestCase):
    def test_recovers_three_blobs(self):
        rng = random.Random(42)
        pts, centers = three_blobs(rng, per=20)
        km = KMedoids(n_clusters=3, metric="euclidean").fit(pts)

        self.assertEqual(len(km.labels_), len(pts))
        self.assertEqual(set(km.labels_), {0, 1, 2})
        # every group of 20 consecutive points should share one label
        groups = [km.labels_[i * 20:(i + 1) * 20] for i in range(3)]
        for g in groups:
            self.assertEqual(len(set(g)), 1)
        self.assertEqual(len({g[0] for g in groups}), 3)

    def test_cost_matches_manual_recompute(self):
        rng = random.Random(0)
        pts, _ = three_blobs(rng)
        km = KMedoids(n_clusters=3).fit(pts)
        from clara.distance import euclidean

        manual = 0.0
        for p in pts:
            manual += min(euclidean(p, m) for m in km.medoids_)
        self.assertAlmostEqual(manual, km.cost_, places=9)

    def test_swap_never_increases_cost_vs_build(self):
        rng = random.Random(7)
        pts, _ = three_blobs(rng)
        from clara.distance import euclidean, pairwise_matrix

        D = pairwise_matrix(pts, euclidean)
        build_medoids = _build_init(D, 3)
        _, build_cost = _assign(D, build_medoids)
        km = KMedoids(n_clusters=3).fit(pts)
        self.assertLessEqual(km.cost_, build_cost + 1e-9)

    def test_predict_new_points(self):
        rng = random.Random(3)
        pts, centers = three_blobs(rng, per=20)
        km = KMedoids(n_clusters=3).fit(pts)
        preds = km.predict([list(c) for c in centers])
        self.assertEqual(len(set(preds)), 3)
        # a point right on a training point gets that point's label
        self.assertEqual(km.predict([pts[0]])[0], km.labels_[0])

    def test_precomputed_matrix(self):
        D = [
            [0.0, 2.0, 100.0, 101.0],
            [2.0, 0.0, 101.0, 100.0],
            [100.0, 101.0, 0.0, 3.0],
            [101.0, 100.0, 3.0, 0.0],
        ]
        km = KMedoids(n_clusters=2, precomputed=True).fit(D)
        self.assertEqual(km.labels_[0], km.labels_[1])
        self.assertEqual(km.labels_[2], km.labels_[3])
        self.assertNotEqual(km.labels_[0], km.labels_[2])
        with self.assertRaises(ValueError):
            km.predict([[0.0]])

    def test_random_init_is_seeded(self):
        rng = random.Random(11)
        pts, _ = three_blobs(rng)
        a = KMedoids(n_clusters=3, init="random", random_state=5).fit(pts)
        b = KMedoids(n_clusters=3, init="random", random_state=5).fit(pts)
        self.assertEqual(a.medoid_indices_, b.medoid_indices_)

    def test_k_equals_n(self):
        pts = [[0.0], [1.0], [2.0]]
        km = KMedoids(n_clusters=3).fit(pts)
        self.assertAlmostEqual(km.cost_, 0.0)
        self.assertEqual(sorted(km.labels_), [0, 1, 2])

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            KMedoids(n_clusters=0)
        with self.assertRaises(ValueError):
            KMedoids(n_clusters=2, init="bogus")
        with self.assertRaises(ValueError):
            KMedoids(n_clusters=5).fit([[0.0], [1.0]])


if __name__ == "__main__":
    unittest.main()
