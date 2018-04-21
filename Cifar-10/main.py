import cifar10
from cifar10 import IMG_SIZE, NUM_CHANNELS, NUM_CALSSES


import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

class_names = cifar10.load_class_names()
print("class names : ", class_names)

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test    = cifar10.load_test_data()

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))


################ Param #############
batch_size = 1000
steps = 20000
drop_out_rate = 0.4
learning_rate = 0.001
#####################

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, cifar10.IMG_SIZE, cifar10.IMG_SIZE, cifar10.NUM_CHANNELS])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu
    )

    # Size of pool1 ??
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)
    dense1 = tf.layers.dense(inputs=pool2, units=256)
    dense2 = tf.layers.dense(inputs=dense1, units=128)

    logits = tf.layers.dense(inputs=dense2, units=10)
    flat1 = tf.layers.flatten(inputs=logits)
    predictions = {
        "classes" : tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(labels, 1), logits=flat1)
    
    # TRAINING
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    # Evaluation
    predictions = {
    "classes": tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"]
            )
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
        )
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )

def main(unused_argv):
    training_data, _, training_label = cifar10.load_training_data()
    test_data, _, test_label = cifar10.load_test_data()
    print(test_label)
    cifar_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='./model'
    )
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100
    )
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_data},
        y=training_label,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
    )
    cifar_classifier.train(
        input_fn=train_input_fn,
        steps=steps
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":test_data},
        y=test_label,
        num_epochs=1,
        shuffle=False
    )
    test_result = cifar_classifier.evaluate(input_fn=eval_input_fn)
    print(test_result)

if __name__ == "__main__":
    tf.app.run()