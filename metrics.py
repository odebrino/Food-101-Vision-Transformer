"""Classification scores for the ten Food-101 project labels."""

from operator import index


def classification_scores(true_labels, predicted_labels):
    """Return top-1 accuracy and macro F1 as fractions for paired class IDs."""
    actual = [0] * 10
    predicted = [0] * 10
    correct = [0] * 10

    for true, guess in zip(true_labels, predicted_labels, strict=True):
        true, guess = index(true), index(guess)
        if not (0 <= true < 10 and 0 <= guess < 10):
            raise ValueError("Expected class IDs from 0 to 9.")
        actual[true] += 1
        predicted[guess] += 1
        if true == guess:
            correct[true] += 1

    total = sum(actual)
    if total == 0:
        raise ValueError("At least one labeled prediction is required.")

    f1_sum = 0.0
    for label in range(10):
        denominator = actual[label] + predicted[label]
        if denominator:
            f1_sum += 2 * correct[label] / denominator
    return {"accuracy": sum(correct) / total, "macro_f1": f1_sum / 10}
