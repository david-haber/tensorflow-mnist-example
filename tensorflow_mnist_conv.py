from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import yaml

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def get_hyperparameters(job_id):
    # Update file name with correct path
    with open("hyperparams.yml", 'r') as stream:
        hyper_param_set = yaml.load(stream)
    print("\nHypermeter set for job_id: ", job_id)
    print("------------------------------------")
    print(hyper_param_set[job_id - 1]["hyperparam_set"])
    print("------------------------------------\n")

    return hyper_param_set[job_id - 1]["hyperparam_set"]


def mnist_net(batch_input):
    # Conv layer
    x = tf.layers.conv2d(batch_input, 16, (3, 3))
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding="same")

    # FC layer
    x = tf.reshape(x, [-1, 13 * 13 * 16])
    x = tf.layers.dense(x, 10)

    return x


def main(log_dir):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Load hyperparameters
    job_id = int(os.environ['JOB_ID'])
    hyperparams = get_hyperparameters(job_id)

    # Define input/output placeholders
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    # Reshape input
    x_image = tf.reshape(x, [-1, 28, 28, 1], name="input_image")

    with tf.variable_scope("mnist_net"):
        # Conv layer
        out = mnist_net(x_image)

        # Define loss function
    with tf.name_scope("loss"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out))

    # Another metric
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Define optimizer
    with tf.name_scope("optimizer"):
        train_step = tf.train.AdamOptimizer(learning_rate=hyperparams["learning_rate"]).minimize(cross_entropy)

    # Define summaries
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    saver_var_list = [var for var in tf.trainable_variables() if var.name.startswith("mnist_net")]
    saver = tf.train.Saver(var_list=saver_var_list)

    sv = tf.train.Supervisor(logdir=log_dir, summary_op=None, saver=saver)

    with sv.managed_session() as sess:

        # Main training loop
        for step in range(500):
            batch_xs, batch_ys = mnist.train.next_batch(32)
            _, summary = sess.run([train_step, merged], feed_dict={x: batch_xs, y: batch_ys})
            sv.summary_writer.add_summary(summary, step)

            if step % 10 == 0:
                print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

        save_path = saver.save(sess, os.path.join(log_dir, "final_model.ckpt"), global_step=sv.global_step)
        print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    title = __file__.split(".")[0]
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join("logs", now + "-" + title)
    os.makedirs(log_dir)

    main(log_dir)
