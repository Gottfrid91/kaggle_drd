
import os
import tensorflow as tf

IMAGE_SIZE = 462
data_dir = '/home/olle/PycharmProjects/kaggle_drd/validation'
batch_size = 1
# Global constants describing the Diabetic Retinopath Detection data set.
NUM_CLASSES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500 # was set from # 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3500
CAPACITY = 200 #number of elements to queue



filenames = [os.path.join(data_dir, 'data_batch_0.bin')]
for f in filenames:
    if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)

# #sppecifying angles for images to be rotated by
# number_of_samples =

# Create a queue that produces the filenames to read.
filename_queue = tf.train.string_input_producer(filenames)
# Read examples from files in the filename queue
print("the filename queue is {}".format(filename_queue))


class SVHNRecord(object):
    pass


result = SVHNRecord()

# Dimensions of the images in the SVHN dataset.
# See http://ufldl.stanford.edu/housenumbers/ for a description of the
# input format.
result.height = 512
result.width = 512
result.depth = 3

reader = tf.TFRecordReader()
result.key, value = reader.read(filename_queue)
value = tf.parse_single_example(
    value,
    # Defaults are not specified since both keys are required.
    features={
        'image_raw': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'label': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        'image_name': tf.FixedLenFeature(shape=[], dtype=tf.string)
    })

# Convert from a string to a vector of uint8 that is record_bytes long.
record_bytes = tf.decode_raw(value['image_raw'], tf.uint8)
#print("THE ROCERD RAW BAYTES HAVE:{}".format(record_bytes.get_shape()))
record_bytes = tf.reshape(record_bytes, [result.height, result.width, 3])
name = tf.cast(value['image_name'], tf.string)
# record_bytes.set_shape([32*32*3])
# # Build an initialization operation to run below.
init = tf.global_variables_initializer()
# Start running operations on the Graph.
sess = tf.Session()
# sess.run(init)

# Start the queue runners.
tf.train.start_queue_runners(sess=sess)
names = []
for i in range(0,3512):
    im_name = sess.run(name)
    names.append(im_name)

names_set = set(names)
print len(names_set)
print(len(names))