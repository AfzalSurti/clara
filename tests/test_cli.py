import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from clara.cli import run
from clara.dataset import make_blobs


class CliTestBase(unittest.TestCase):
    def make_csv(self, with_header=True, label_col=False):
        X, y = make_blobs(
            n_samples=200,
            centers=[(0, 0), (20, 0), (0, 20)],
            cluster_std=1.0,
            random_state=1,
        )
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        self.addCleanup(os.remove, path)
        header = ["x", "y"] if with_header else None
        if label_col:
            rows = [row + [f"c{lab}"] for row, lab in zip(X, y)]
            hdr = (header + ["label"]) if header else None
            lines = []
            if hdr:
                lines.append(",".join(hdr))
            for r in rows:
                lines.append(",".join(str(v) for v in r))
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write("\n".join(lines) + "\n")
        else:
            lines = []
            if header:
                lines.append(",".join(header))
            for r in X:
                lines.append(",".join(str(v) for v in r))
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write("\n".join(lines) + "\n")
        return path

    def call(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = run(argv)
        return code, out.getvalue(), err.getvalue()


class TestCli(CliTestBase):
    def test_clara_summary_runs(self):
        path = self.make_csv()
        code, out, err = self.call([path, "-k", "3", "--seed", "0"])
        self.assertEqual(code, 0, err)
        self.assertIn("algorithm      : clara", out)
        self.assertIn("cluster sizes", out)
        self.assertIn("medoid 0", out)

    def test_pam_algorithm(self):
        path = self.make_csv()
        code, out, err = self.call([path, "-k", "3", "-a", "pam", "--seed", "0"])
        self.assertEqual(code, 0, err)
        self.assertIn("algorithm      : pam", out)

    def test_quiet_prints_only_labels(self):
        path = self.make_csv()
        code, out, err = self.call([path, "-k", "3", "--seed", "0", "-q"])
        self.assertEqual(code, 0, err)
        tokens = out.split()
        self.assertEqual(len(tokens), 200)
        self.assertTrue(all(t in {"0", "1", "2"} for t in tokens))

    def test_output_file_written(self):
        path = self.make_csv()
        out_path = path + ".clustered.csv"
        self.addCleanup(lambda: os.path.exists(out_path) and os.remove(out_path))
        code, out, err = self.call([path, "-k", "3", "--seed", "0", "-o", out_path])
        self.assertEqual(code, 0, err)
        self.assertTrue(os.path.exists(out_path))
        with open(out_path, encoding="utf-8") as fh:
            first = fh.readline().strip()
        self.assertEqual(first, "x,y,cluster")

    def test_silhouette_flag(self):
        path = self.make_csv()
        code, out, err = self.call([path, "-k", "3", "--seed", "0", "--silhouette"])
        self.assertEqual(code, 0, err)
        self.assertIn("silhouette      :", out)

    def test_label_column_excluded_from_features(self):
        path = self.make_csv(label_col=True)
        code, out, err = self.call(
            [path, "-k", "3", "--seed", "0", "--label-column", "-1"]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("2 features", out)

    def test_bad_k_errors(self):
        path = self.make_csv()
        code, out, err = self.call([path, "-k", "0"])
        self.assertEqual(code, 2)
        self.assertIn("-k must be between", err)

    def test_missing_file_errors(self):
        code, out, err = self.call(["does-not-exist.csv", "-k", "2"])
        self.assertEqual(code, 2)
        self.assertIn("clara:", err)

    def test_manhattan_metric_runs(self):
        path = self.make_csv()
        code, out, err = self.call([path, "-k", "3", "-m", "manhattan", "--seed", "0"])
        self.assertEqual(code, 0, err)
        self.assertIn("metric         : manhattan", out)


if __name__ == "__main__":
    unittest.main()
