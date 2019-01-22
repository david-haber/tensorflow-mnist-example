from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def mnist_net(batch_input):
    # Conv layer
    with tf.variable_scope("conv"):
        filters = tf.get_variable("filters", initializer=tf.truncated_normal(shape=[3, 3, 1, 16], stddev=0.1))
        conv = tf.nn.conv2d(batch_input, filters, strides=[1, 1, 1, 1], padding="VALID")
        act = tf.nn.relu(conv)
        max_pool = tf.nn.max_pool(act, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

    # FC layer
    with tf.variable_scope("fc"):
        W_fc = tf.get_variable("W", initializer=tf.truncated_normal([13 * 13 * 16, 10], stddev=0.1))
        b_fc = tf.get_variable("b", initializer=tf.constant(0.0, shape=[10]))

        pool_flat = tf.reshape(max_pool, [-1, 13 * 13 * 16])
        out_fc = tf.nn.softmax(tf.matmul(pool_flat, W_fc) + b_fc)

    return out_fc


def main(log_dir):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

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
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))

    # Another metric
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Define optimizer
    with tf.name_scope("optimizer"):
        train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    # Define summaries
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    saver_var_list = [var for var in tf.trainable_variables() if var.name.startswith("mnist_net")]
    saver = tf.train.Saver(var_list=saver_var_list)

    sv = tf.train.Supervisor(logdir=log_dir, summary_op=None, saver=saver)

    with sv.managed_session() as sess:

        # Main training loop
        for step in range(10000):
            batch_xs, batch_ys = mnist.train.next_batch(32)
            _, summary = sess.run([train_step, merged], feed_dict={x: batch_xs, y: batch_ys})
            sv.summary_writer.add_summary(summary, step)

            if step % 100 == 0:
                print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

        save_path = saver.save(sess, os.path.join(log_dir, "final_model.ckpt"), global_step=sv.global_step)
        print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    title = __file__.split(".")[0]
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join("logs", now + "-" + title)
    os.makedirs(log_dir)

    main(log_dir)
