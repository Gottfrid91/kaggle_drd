# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import drd
import t_sne
import evaluation_functions as ef
import matplotlib.pyplot as plt
from resnet_model_offic import Model

import model_2


parser = drd.parser

parser.add_argument('--eval_dir', type=str,
                    default='./tf_offic_weights/',
                    help="""Directory where to write event logs.""")
parser.add_argument('--eval_data', type=str, default='/media/olle/Seagate Expansion Drive/DRD_master_thesis_olle_holmberg/kaggle/test_zip_files/records/',
                    help="""Either 'test' or 'train_eval'.""")
parser.add_argument('--checkpoint_dir', type=str,
                    default='./tf_offic_weights/',
                    help="""Directory where to read model checkpoints.""")
parser.add_argument('--eval_interval_secs', type=int, default=60,
                    help="""How often to run the eval.""")
parser.add_argument('--num_examples', type=int, default=1000,
                    help="""Number of examples to run.""")
parser.add_argument('--run_once', type=bool, default=True,
                    help='Whether to run eval only once.')
parser.add_argument('--n_residual_blocks', type=int, default=5,
                    help='Number of residual blocks in network')

weight_decay = 0.0001
momentum=0.9
lr = 0.001

def eval_once(saver, summary_writer, top_k_op, summary_op, logits, images, labels, prediction, acc):
    """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op
  """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt)
            print(ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            #initializa un-intilized variables
            uninitialized_vars = []
            for var in tf.global_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    print("not init")
                    print(var)
                    uninitialized_vars.append(var)

            # create init op for the still unitilized variables
            init_new_vars_op = tf.variables_initializer(uninitialized_vars)
            sess.run(init_new_vars_op)

        else:
            print('No checkpoint file found')
            return

        # lists to append results for visualizations
        final_layer = []
        class_pred = []
        class_labels = []
        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            avg_accuracy = []
            step = 0
            while step < num_iter and not coord.should_stop():
                if step % 100 == 0:
                    print("The step is {}".format(step))
                predictions, f_layer, cls_labels, cls_prediction, a = sess.run([top_k_op, logits, labels, prediction, acc])
                # book keep prediction, logits, labels
                class_pred.append(cls_prediction)
                final_layer.append(f_layer)
                class_labels.append(cls_labels)
                avg_accuracy.append(a)

                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
            print("avg accuracy is: {}".format(np.mean(avg_accuracy)))
            # convert bookkeeping to numpy for helper function
            class_labels = np.concatenate(class_labels, axis=0)
            final_layer = np.concatenate(final_layer, axis=0)
            class_pred = np.concatenate(class_pred, axis=0).reshape(len(class_pred), 1)

            # here insert the TSNET visualization
            t_sne.plot_embedding(t_sne.t_sne_fit(final_layer), sess.run(images),
                                 class_labels,
                                 iter, title="Final layer representation")

            #ef.plot_confusion_matrix(class_pred, class_labels, 5)
            TP, FP, TN, FN = ef.perf_measure(class_labels,class_pred)
            ef.plot_confusion_matrix_2(cls_pred=class_pred,cls_true=class_labels)

            # Plot non-normalized confusion matrix
            plt.figure()
            ef.plot_confusion_matrix_2(class_pred, class_labels)

            # Plot normalized confusion matrix
            plt.figure()
            ef.plot_confusion_matrix_2(class_pred, class_labels, normalize=False)

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        eval_data = FLAGS.eval_data

        images, labels = drd.test_inputs(eval_data, batch_size=1)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        model = Model(resnet_size='50', data_format=None, resnet_version=2,
                            dtype=tf.float32)
        # mode is training param for batch norm
        logits = model(images, training=False,is_batch_norm = True)
        print(logits.get_shape(), labels.get_shape())

        # logits1= drd.inference(images, FLAGS.n_residual_blocks)
        # logits = model_2.inference(images, n=4, reuse=tf.AUTO_REUSE)
        # val_logits = model_2.inference(images, n=4, reuse=tf.AUTO_REUSE)
        # logits = drd.resnet_v1_50(images, training=True)
        # val_logits = drd.resnet_v1_50(val_images, training = False)

        # softmx logits
        soft_max_logits = tf.nn.softmax(logits)
        # calculate predictions
        predictions = tf.cast(tf.argmax(soft_max_logits, axis=1), tf.int32)
        top_k_op = tf.nn.in_top_k(soft_max_logits, labels, 1)

        # ops for batch accuracy calcultion
        correct_prediction = tf.equal(predictions, labels)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Calculate loss, which includes softmax cross entropy and L2 regularization.
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels)

        # Create a tensor named cross_entropy for logging purposes.
        tf.identity(cross_entropy, name='cross_entropy')
        tf.summary.scalar('cross_entropy', cross_entropy)

        # If no loss_filter_fn is passed, assume we want the default behavior,
        # which is that batch_normalization variables are excluded from loss.
        def exclude_batch_norm(name):
            return 'batch_normalization' not in name

        loss_filter_fn = None or exclude_batch_norm

        # Add weight decay to the loss.
        l2_loss = weight_decay * tf.add_n(
            # loss is computed using fp32 for numerical stability.
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
             if loss_filter_fn(v.name)])
        tf.summary.scalar('l2_loss', l2_loss)
        loss = cross_entropy + l2_loss

        # list of lr decay factors
        lr_decay_factors = [1, 0.1, 0.01, 0.001, 0.0001]
        learning_rate = 0.001
        # Create a tensor named learning_rate for logging purposes
        # Restore the moving average version of the learned variables for eval.
        saver = tf.train.Saver()

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op, logits, images, labels, predictions, acc)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    evaluate()


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()
