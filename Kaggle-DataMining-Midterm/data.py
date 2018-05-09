import csv
import numpy as np
def load_training_data(filepath):
    features = []
    labels = []
    with open(filepath) as train_file:
        train_data = csv.reader(train_file, quotechar = ',')
        for row in train_data:
            features.append(row[1:129])
            labels.append(row[129])
    features_array = np.array(features, dtype=np.float32)
    labels_array = np.array(labels, dtype=int)
    labels_array = labels_array - 1
    print(features_array.shape, labels_array.shape)

    labels_array = one_hot_encoded(labels_array)
    print(features_array.shape, labels_array.shape)

    # check NaN
    assert not np.any(np.isnan(features_array))
    assert not np.any(np.isnan(labels_array))
    return features_array, labels_array

def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]
