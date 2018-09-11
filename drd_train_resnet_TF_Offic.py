"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from datetime import datetime
import os.path
import time
import unicodedata

import numpy as np
import tensorflow as tf
import sys
import model_2
from tensorflow.contrib import slim
from resnet_model_offic import Model
import drd

#setting params for optimization
weight_decay = 0.0001
momentum=0.9
lr = 0.001
NUM_IMAGES = 28000

#write no pyc files
sys.dont_write_bytecode = True

parser = drd.parser

parser.add_argument('--save_dir', type=str, default='./tf_offic_weights/',
                    help='Directory where to write event logs and checkpoint.')

parser.add_argument('--pre_trained_dir', type=str, default='./output/pre_weights/inference_2Blocks',
                    help='Directory where to write event logs and checkpoint.')

parser.add_argument('--max_steps', type=int, default=20000000,
                    help='Number of batches to run.')

parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

parser.add_argument('--log_frequency', type=int, default=10,
                    help='How often to log results to the console.')
def train():
    #for which data set to use
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name= 'global_step')

        # Get images and labels
        images, labels = drd.distorted_inputs()
        #get validation data
        val_images, val_labels = drd.inputs(False)
        #get drop out probability
        print(images.get_shape(), val_images.get_shape())

        # Build a Graph that computes the logits predictions from the
        # inference model.
        model = Model(resnet_size='50', data_format=None, resnet_version=2,
                            dtype=tf.float32)
        #mode is training param for batch norm
        logits = model(images, training =True,is_batch_norm=True)
        val_logits = model(val_images, training = False,is_batch_norm=True)
        print(logits.get_shape(),labels.get_shape())

        #logits1= drd.inference(images, FLAGS.n_residual_blocks)
        #logits = model_2.inference(images, n=4, reuse=tf.AUTO_REUSE)
        #val_logits = model_2.inference(images, n=4, reuse=tf.AUTO_REUSE)
        #logits = drd.resnet_v1_50(images, training=True)
        #val_logits = drd.resnet_v1_50(val_images, training = False)

        #softmx logits
        soft_max_logits = tf.nn.softmax(logits)
        soft_max_logits_val = tf.nn.softmax(val_logits)
        # calculate predictions
        predictions = tf.cast(tf.argmax(soft_max_logits, axis=1), tf.int32)
        val_predictions = tf.cast(tf.argmax(soft_max_logits_val, axis=1), tf.int32)


        # ops for batch accuracy calcultion
        correct_prediction = tf.equal(predictions, labels)
        val_correct_prediction = tf.equal(val_predictions, labels)

        batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        val_batch_accuracy = tf.reduce_mean(tf.cast(val_correct_prediction, tf.float32))

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
        global_step = tf.train.get_or_create_global_step()

        #list of lr decay factors
        lr_decay_factors =[1, 0.1,0.01,0.001,0.0001]
        learning_rate = 0.00001
        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum
        )

        minimize_op = optimizer.minimize(loss, global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
        # calculate training accuracy
        # Calculate loss.
        #loss = drd.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        #train_op = drd.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        #variables = slim.get_variables_to_restore()
        #variables_to_restore = [v for v in variables if not v.name.split('/')[-1] != 'weights:0']
        # Add ops to save and restore all the variables.
        #saver_pre = tf.train.Saver(variables_to_restore[0:-2])  # exclude logits layer
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        # sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.save_dir, sess.graph)

        step_start = 0
        try:
            ####Trying to find last checkpoint file fore full final model exist###
            print("Trying to restore last checkpoint ...")
            save_dir = FLAGS.save_dir
            # Use TensorFlow to find the latest checkpoint - if any.
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
            # Try and load the data in the checkpoint.
            saver.restore(sess, save_path=last_chk_path)

            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_chk_path)
            # get the step integer from restored path to start step from there
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

        except:
            # If all the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore any checkpoints. Initializing variables instead.")
            sess.run(init)

        accuracy_dev = []
        val_accuracy_dev = []
        step_start = 0
        for step in range(step_start, FLAGS.max_steps):
            start_time = time.time()
            #run train op
            _, loss_value, accuracy, gs= sess.run([train_op, loss, batch_accuracy, global_step])

            #setting up a learning rate decay scheme
            if ((gs*FLAGS.batch_size)/NUM_IMAGES) == 30:
                learning_rate = learning_rate*lr_decay_factors[1]
            if ((gs * FLAGS.batch_size) / NUM_IMAGES) == 60:
                learning_rate = learning_rate * lr_decay_factors[2]
            if ((gs * FLAGS.batch_size) / NUM_IMAGES) == 90:
                learning_rate = learning_rate * lr_decay_factors[3]
            if ((gs * FLAGS.batch_size) / NUM_IMAGES) == 120:
                learning_rate = learning_rate * lr_decay_factors[4]

            accuracy_dev.append(accuracy)
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                val_acc = sess.run([val_batch_accuracy])
                val_accuracy_dev.append(val_acc)

                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f, avg_batch_accuracy = %.2f, (%.1f examples/sec; %.3f '
                              'sec/batch), validation accuracy %.2f')
                # take averages of all the accuracies from the previous bathces
                print(format_str % (datetime.now(), step, loss_value, np.mean(accuracy_dev),
                                    examples_per_sec, sec_per_batch, np.mean(val_accuracy_dev)))

            if step % 10 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                #set paths and saving ops for the full and sub_network
                checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
                #pre_trained_path = os.path.join(FLAGS.pre_trained_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                #saver_30.save(sess, pre_trained_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    train()

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()