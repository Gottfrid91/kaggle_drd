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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import tensorflow as tf

IMAGE_SIZE = 256

# Global constants describing the Diabetic Retinopath Detection data set.
NUM_CLASSES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 35000 # was set from # 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3500
CAPACITY = 200 #number of elements to queue

def read_svhn(filename_queue):
    """Reads and parses examples from SVHN data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
      filename_queue: A queue of strings with the filenames to read from.

    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class SVHNRecord(object):
        pass

    result = SVHNRecord()

    # Dimensions of the images in the SVHN dataset.
    # See http://ufldl.stanford.edu/housenumbers/ for a description of the
    # input format.
    result.height = 256
    result.width = 256
    result.depth = 3

    reader = tf.TFRecordReader()
    result.key, value = reader.read(filename_queue)
    value = tf.parse_single_example(
        value,
        # Defaults are not specified since both keys are required.
        features={
            'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'label': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        })

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value['image'], tf.float32)
    print("THE ROCERD RAW BAYTES HAVE:{}".format(record_bytes.get_shape()))
    # record_bytes.set_shape([32*32*3])
    record_bytes = tf.reshape(record_bytes, [result.height, result.width, 3])
    print("record bytes::::: ", record_bytes)
    # Store our label to result.label and convert to int32
    result.label = tf.cast(value['label'], tf.int32)
    result.uint8image = record_bytes

    return result


def _generate_image_and_label_batch(image, label,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=CAPACITY+3*batch_size,
        min_after_dequeue=CAPACITY)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=CAPACITY+3*batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)
  tf.summary.scalar('label', label)

  return images, tf.reshape(label_batch, [batch_size])

def get_filenames(is_training, data_dir):
  if is_training:
    return [os.path.join(data_dir, 'retinopathy_tr.tfrecords')]
  else:
    return [os.path.join(data_dir, 'retinopathy_vl.tfrecords')]

def get_test_filenames(data_dir):
    return [os.path.join(data_dir, '_test.tfrecords')]
def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for SVHN training using the Reader ops.

    Args:
      data_dir: Path to the SVHN data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    is_training = True
    filenames = get_filenames(is_training, data_dir)

    # #sppecifying angles for images to be rotated by
    # number_of_samples =

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)
    # Read examples from files in the filename queue
    print("the filename queue is {}".format(filename_queue))
    read_input = read_svhn(filename_queue)
    image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE
    NUM_CHANNELS = 3
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, height + 8, width + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [height, width, NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    angles = tf.random_uniform([1], -15, 15, dtype=tf.float32, seed=0)
    image = tf.contrib.image.rotate(image, angles * math.pi / 360, interpolation='NEAREST', name=None)

    print('Filling queue with %d DRD images before starting to train. '
          'This will take a few minutes.' % CAPACITY)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(image, read_input.label, batch_size,shuffle=True)


def inputs(eval_data, batch_size, data_dir):
    """Construct input for SVHN evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path
       to the SVHN data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    is_training = False
    filenames = get_filenames(is_training, data_dir)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    print("the filename queue is {}".format(filename_queue))
    read_input = read_svhn(filename_queue)
    image = tf.cast(read_input.uint8image, tf.float32)

    print('Filling queue with %d DRD images before starting to train. '
          'This will take a few minutes.' % CAPACITY)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(image, read_input.label, batch_size,shuffle=False)

def test_inputs(data_dir, batch_size=1):
    """Construct input for SVHN evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path
       to the SVHN data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = get_test_filenames(data_dir)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    print("the filename queue is {}".format(filename_queue))
    read_input = read_svhn(filename_queue)
    image = tf.cast(read_input.uint8image, tf.float32)

    print('Filling queue with %d DRD images before starting to train. '
          'This will take a few minutes.' % CAPACITY)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(image, read_input.label, batch_size,shuffle=False)
