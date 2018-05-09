import numpy as np
import tensorflow as tf
import data

tf.logging.set_verbosity(tf.logging.INFO)

# Parameter goes here
batch_size = 100
steps = 200
drop_out_rate = 0.2
learning_rate = 0.0001
def nn_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 128])

    dense_1 = tf.layers.dense(inputs=input_layer, units=128, activation=tf.nn.relu, use_bias=True)
    # dense_2 = tf.layers.dense(inputs=dense_1, units=128, activation=tf.nn.relu, use_bias=True)
    dense_3 = tf.layers.dense(inputs=dense_1, units=32, activation=tf.nn.relu, use_bias=True)


    dropout = tf.layers.dropout(
        inputs=dense_3, rate=drop_out_rate, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=6)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Calculate loss
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(labels, 1), logits=logits)
    loss = tf.losses.softmax_cross_entropy(labels, logits)

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
                labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"]
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
    # Load training and eval data
    train_data, train_labels = data.load_training_data('train.csv')

    kaggle_classifier = tf.estimator.Estimator(
        model_fn=nn_model_fn, model_dir='./model'
    )
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100
    )
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
    )
    kaggle_classifier.train(
        input_fn=train_input_fn,
        steps=steps)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        num_epochs=1,
        shuffle=False)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        shuffle=False
    )
    eval_results = kaggle_classifier.evaluate(input_fn=eval_input_fn)
    predict_results = kaggle_classifier.predict(input_fn=predict_input_fn)
    for item in predict_results:
        print(item)
    print(eval_results)

if __name__ == "__main__":
  tf.app.run()