import unittest

import pandas as pd

from theval.collect import extract_summary_value
from theval.scoring import compute_final_scores, score_csv


class ScoringTest(unittest.TestCase):
    def test_final_score_is_average_normalized_score(self):
        df = pd.DataFrame(
            {
                "Model": ["GT", "Method"],
                "Global aesthetic": [10.0, 12.0],
                "Mouth quality": [10.0, 5.0],
                "Face quality": [10.0, 10.0],
                "Lip dynamics": [10.0, 15.0],
                "Head motion dynamics": [10.0, 10.0],
                "Eyebrow dynamics": [10.0, 5.0],
                "Silent lip stability": [10.0, 10.0],
                "Lip sync": [10.0, 0.0],
            }
        ).set_index("Model")

        scores = compute_final_scores(df)

        self.assertAlmostEqual(scores.loc["GT", "Final score"], 1.0)
        self.assertAlmostEqual(scores.loc["Method", "Final score"], 0.6625)
        self.assertEqual(scores.loc["GT", "Rank"], 1)
        self.assertEqual(scores.loc["Method", "Rank"], 2)

    def test_metric_column_aliases_are_accepted(self):
        df = pd.DataFrame(
            {
                "Model": ["GT", "Method"],
                "global_aesthetic": [1.0, 2.0],
                "mouth_quality": [1.0, 1.0],
                "face_quality": [1.0, 1.0],
                "std_over_time": [1.0, 1.0],
                "head_motion": [1.0, 1.0],
                "micro_expression_intensity": [1.0, 1.0],
                "silent_mad": [1.0, 1.0],
                "mean_difference": [1.0, 1.0],
            }
        ).set_index("Model")

        scores = compute_final_scores(df)

        self.assertAlmostEqual(scores.loc["Method", "Final score"], 0.875)

    def test_optional_weights(self):
        df = pd.DataFrame(
            {
                "Model": ["GT", "Method"],
                "Global aesthetic": [1.0, 2.0],
                "Mouth quality": [1.0, 2.0],
                "Face quality": [1.0, 1.0],
                "Lip dynamics": [1.0, 1.0],
                "Head motion dynamics": [1.0, 1.0],
                "Eyebrow dynamics": [1.0, 1.0],
                "Silent lip stability": [1.0, 1.0],
                "Lip sync": [1.0, 1.0],
            }
        ).set_index("Model")

        scores = compute_final_scores(df, weights={"Global aesthetic": 2, "Mouth quality": 0})

        self.assertAlmostEqual(scores.loc["Method", "Final score"], 0.75)

    def test_gt_reference_file_scores_gt_as_perfect(self):
        scores = score_csv("examples/gt_metrics.csv")

        self.assertEqual(list(scores.index), ["GT"])
        self.assertAlmostEqual(scores.loc["GT", "Final score"], 1.0)


class CollectTest(unittest.TestCase):
    def test_extract_summary_value(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "global_aesthetic.txt"
            output_file.write_text("\n=== Summary ===\nOverall Average Aesthetics: 4.1234\n")

            value = extract_summary_value("Global aesthetic", output_file)

        self.assertAlmostEqual(value, 4.1234)


if __name__ == "__main__":
    unittest.main()
