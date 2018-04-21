import numpy as np
import pickle
import matplotlib.pyplot as plt
import os


data_path = "./dataset/"
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# Width and height of each image
IMG_SIZE = 32

# Number of channel
NUM_CHANNELS = 3

IMG_SIZE_FLAT = IMG_SIZE * IMG_SIZE * NUM_CHANNELS

NUM_CALSSES = 10

NUM_FILES_TRAIN = 5

IMAGES_PER_FILE = 10000

NUM_IMAGES_TRAIN = NUM_FILES_TRAIN * IMAGES_PER_FILE

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

def get_file_path(filename=""):
    return os.path.join(data_path, "cifar-10-batches-py/", filename)

def unpickle(filename):
    file_path = get_file_path(filename)

    print("Loading data : " + file_path)

    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    
    return data

def convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Normalize
    raw_float = np.array(raw, dtype=float) / 255.0

    images = raw_float.reshape([-1, NUM_CHANNELS, IMG_SIZE, IMG_SIZE])

    # Reorder the index of the array
    images = images.transpose([0, 2, 3, 1])

    return images

def load_data(filename):

    data = unpickle(filename)

    raw_images = data[b'data']

    # Get the class-number for each image. Convert to numpy-array
    cls = np.array(data[b'labels'])

    # Convert the images
    images = convert_images(raw_images)

    return images, cls

def load_class_names():
    raw = unpickle(filename="batches.meta")[b'label_names']

    names = [x.decode('utf-8') for x in raw]

    return names

def load_training_data():
    """
    Load all the training-data for the CIFAR-10 data-set.
    The data-set is split into 5 data-files which are merged here.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    images = np.zeros(shape=[NUM_IMAGES_TRAIN, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], dtype=np.float32)
    cls = np.zeros(shape=[NUM_IMAGES_TRAIN], dtype=int)

    # Begin-index for the current batch
    begin = 0

    for i in range(NUM_FILES_TRAIN):
        images_batch, cls_batch = load_data(filename="data_batch_" + str(i + 1))

        num_images = len(images_batch)

        # Calculate the End-Index
        end = begin + num_images

        images[begin:end, :] = images_batch

        cls[begin:end] = cls_batch

        # Update Begin-Index
        begin = end
    
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=NUM_CALSSES)

def load_test_data():
    images, cls = load_data(filename="test_batch")
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=NUM_CALSSES)


def plot_images(images, cls_true, class_names, cls_pred=None, smooth=True):

    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3, 3)

    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'
        
        ax.imshow(images[i, :, :, :],
                    interpolation=interpolation)
        
        cls_true_name = class_names[cls_true[i]]

        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            cls_pred_name = class_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
        
        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()

