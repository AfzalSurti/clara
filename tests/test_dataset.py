import os
import tempfile
import unittest

from clara.dataset import _split_csv_line, load_csv, make_blobs, write_labeled_csv


class TestCsvLineSplitter(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(_split_csv_line("1,2,3", ","), ["1", "2", "3"])

    def test_quoted_with_comma(self):
        self.assertEqual(_split_csv_line('a,"b,c",d', ","), ["a", "b,c", "d"])

    def test_doubled_quote_escape(self):
        self.assertEqual(_split_csv_line('"he said ""hi""",x', ","), ['he said "hi"', "x"])

    def test_trailing_empty_field(self):
        self.assertEqual(_split_csv_line("1,2,", ","), ["1", "2", ""])

    def test_alt_delimiter(self):
        self.assertEqual(_split_csv_line("1;2;3", ";"), ["1", "2", "3"])


class TestLoadCsv(unittest.TestCase):
    def _write(self, text):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        self.addCleanup(os.remove, path)
        return path

    def test_auto_detects_header(self):
        path = self._write("x,y\n1,2\n3,4\n")
        X, labels, header = load_csv(path)
        self.assertEqual(header, ["x", "y"])
        self.assertEqual(X, [[1.0, 2.0], [3.0, 4.0]])
        self.assertIsNone(labels)

    def test_auto_detects_no_header(self):
        path = self._write("1,2\n3,4\n")
        X, labels, header = load_csv(path)
        self.assertIsNone(header)
        self.assertEqual(X, [[1.0, 2.0], [3.0, 4.0]])

    def test_label_column_positive_and_negative(self):
        path = self._write("f1,f2,label\n1,2,A\n3,4,B\n")
        X, labels, header = load_csv(path, label_column=-1)
        self.assertEqual(X, [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(labels, ["A", "B"])
        X2, labels2, _ = load_csv(path, label_column=2)
        self.assertEqual(labels2, ["A", "B"])
        self.assertEqual(X2, X)

    def test_ragged_row_raises(self):
        path = self._write("1,2\n3,4,5\n")
        with self.assertRaises(ValueError):
            load_csv(path)

    def test_non_numeric_feature_raises(self):
        path = self._write("1,2\n3,foo\n")
        with self.assertRaises(ValueError):
            load_csv(path)

    def test_blank_lines_skipped(self):
        path = self._write("1,2\n\n3,4\n\n")
        X, _, _ = load_csv(path)
        self.assertEqual(X, [[1.0, 2.0], [3.0, 4.0]])

    def test_round_trip_with_write(self):
        path = self._write("x,y\n0,0\n10,10\n")
        X, _, header = load_csv(path)
        out = path + ".out.csv"
        self.addCleanup(lambda: os.path.exists(out) and os.remove(out))
        write_labeled_csv(out, X, [0, 1], header)
        X2, labels2, header2 = load_csv(out, label_column=-1)
        self.assertEqual(header2, ["x", "y"])  # 'cluster' pulled out as the label
        self.assertEqual(X2, X)
        self.assertEqual(labels2, ["0", "1"])


class TestMakeBlobs(unittest.TestCase):
    def test_shape_and_labels(self):
        X, y = make_blobs(n_samples=100, centers=4, n_features=3, random_state=1)
        self.assertEqual(len(X), 100)
        self.assertEqual(len(y), 100)
        self.assertTrue(all(len(row) == 3 for row in X))
        self.assertEqual(set(y), {0, 1, 2, 3})

    def test_explicit_centers_infer_features(self):
        X, y = make_blobs(n_samples=30, centers=[(0, 0, 0), (9, 9, 9)], random_state=2)
        self.assertEqual(len(X[0]), 3)
        self.assertEqual(set(y), {0, 1})

    def test_reproducible(self):
        a = make_blobs(n_samples=50, centers=3, random_state=5)
        b = make_blobs(n_samples=50, centers=3, random_state=5)
        self.assertEqual(a, b)

    def test_uneven_split_counts_all_samples(self):
        X, y = make_blobs(n_samples=101, centers=3, random_state=0)
        self.assertEqual(len(X), 101)


if __name__ == "__main__":
    unittest.main()
