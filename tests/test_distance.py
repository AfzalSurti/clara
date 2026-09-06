import math
import unittest

from clara import distance


class TestDistanceFunctions(unittest.TestCase):
    def test_euclidean_basic(self):
        self.assertAlmostEqual(distance.euclidean([0, 0], [3, 4]), 5.0)
        self.assertAlmostEqual(distance.euclidean([1, 2, 3], [1, 2, 3]), 0.0)

    def test_sqeuclidean_is_euclidean_squared(self):
        a, b = [1.0, -2.0, 0.5], [4.0, 0.0, -1.5]
        self.assertAlmostEqual(
            distance.sqeuclidean(a, b), distance.euclidean(a, b) ** 2
        )

    def test_manhattan_basic(self):
        self.assertAlmostEqual(distance.manhattan([0, 0], [3, 4]), 7.0)
        self.assertAlmostEqual(distance.manhattan([-1, -1], [1, 1]), 4.0)

    def test_chebyshev_basic(self):
        self.assertAlmostEqual(distance.chebyshev([0, 0], [3, 4]), 4.0)
        self.assertAlmostEqual(distance.chebyshev([1, 9, 2], [1, 1, 1]), 8.0)

    def test_minkowski_matches_special_cases(self):
        a, b = [0.0, 0.0], [3.0, 4.0]
        self.assertAlmostEqual(distance.minkowski(a, b, p=1), distance.manhattan(a, b))
        self.assertAlmostEqual(distance.minkowski(a, b, p=2), distance.euclidean(a, b))
        self.assertAlmostEqual(distance.minkowski(a, b, p=100), distance.chebyshev(a, b), places=1)

    def test_minkowski_rejects_bad_order(self):
        with self.assertRaises(ValueError):
            distance.minkowski([1], [2], p=0.5)

    def test_cosine_orthogonal_and_parallel(self):
        self.assertAlmostEqual(distance.cosine([1, 0], [0, 1]), 1.0)
        self.assertAlmostEqual(distance.cosine([1, 1], [2, 2]), 0.0)
        self.assertAlmostEqual(distance.cosine([1, 0], [-1, 0]), 2.0)

    def test_cosine_zero_vectors(self):
        self.assertAlmostEqual(distance.cosine([0, 0], [0, 0]), 0.0)
        self.assertAlmostEqual(distance.cosine([0, 0], [1, 1]), 1.0)

    def test_length_mismatch_raises(self):
        for fn in (distance.euclidean, distance.manhattan, distance.chebyshev, distance.cosine):
            with self.assertRaises(ValueError):
                fn([1, 2], [1, 2, 3])

    def test_symmetry_and_non_negativity(self):
        pts = [[0.0, 0.0], [1.5, -2.0], [3.0, 3.0], [-4.0, 1.0]]
        for name in ("euclidean", "manhattan", "chebyshev", "cosine"):
            fn = distance.get_distance(name)
            for a in pts:
                for b in pts:
                    d1, d2 = fn(a, b), fn(b, a)
                    self.assertGreaterEqual(d1, 0.0)
                    self.assertAlmostEqual(d1, d2)


class TestResolver(unittest.TestCase):
    def test_get_distance_by_name(self):
        self.assertIs(distance.get_distance("euclidean"), distance.euclidean)
        self.assertIs(distance.get_distance("L1"), distance.manhattan)
        self.assertIs(distance.get_distance("cityblock"), distance.manhattan)

    def test_get_distance_passes_callable_through(self):
        fn = lambda a, b: 0.0
        self.assertIs(distance.get_distance(fn), fn)

    def test_get_distance_unknown(self):
        with self.assertRaises(ValueError):
            distance.get_distance("not-a-distance")


class TestPairwiseMatrix(unittest.TestCase):
    def test_matrix_shape_and_diagonal(self):
        pts = [[0.0], [1.0], [4.0]]
        m = distance.pairwise_matrix(pts, distance.euclidean)
        self.assertEqual(len(m), 3)
        self.assertTrue(all(len(row) == 3 for row in m))
        for i in range(3):
            self.assertEqual(m[i][i], 0.0)

    def test_matrix_values_and_symmetry(self):
        pts = [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]
        m = distance.pairwise_matrix(pts, distance.euclidean)
        self.assertAlmostEqual(m[0][1], 5.0)
        self.assertAlmostEqual(m[0][2], 10.0)
        self.assertAlmostEqual(m[1][2], 5.0)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(m[i][j], m[j][i])


if __name__ == "__main__":
    unittest.main()
