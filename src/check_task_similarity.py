import numpy as np

from generate_data import generate_inputs, generate_labels


X = generate_inputs()

yA, yB_high, yB_medium, yB_low = generate_labels(X)


def agreement(y1, y2):
    return np.mean(y1 == y2)


print("Label agreement with Task A:")
print("B high:  ", agreement(yA, yB_high))
print("B medium:", agreement(yA, yB_medium))
print("B low:   ", agreement(yA, yB_low))