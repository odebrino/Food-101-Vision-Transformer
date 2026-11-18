import unittest

from metrics import classification_scores


class ClassificationScoreTests(unittest.TestCase):
    def test_all_ten_classes_predicted_correctly(self):
        self.assertEqual(classification_scores(range(10), range(10)), {"accuracy": 1.0, "macro_f1": 1.0})

    def test_constant_prediction_on_balanced_labels(self):
        scores = classification_scores(range(10), [0] * 10)
        self.assertAlmostEqual(scores["accuracy"], 0.1)
        self.assertAlmostEqual(scores["macro_f1"], 1 / 55)

    def test_absent_classes_contribute_zero(self):
        scores = classification_scores([0, 0, 0, 1], [0, 0, 1, 1])
        self.assertAlmostEqual(scores["accuracy"], 0.75)
        self.assertAlmostEqual(scores["macro_f1"], (4 / 5 + 2 / 3) / 10)

    def test_invalid_pairs(self):
        for actual, predicted in [([], []), ([1], []), ([-1], [0]), ([0], [10])]:
            with self.subTest(actual=actual, predicted=predicted), self.assertRaises(ValueError):
                classification_scores(actual, predicted)
