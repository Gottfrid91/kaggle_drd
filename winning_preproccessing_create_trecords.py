"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
from PIL import ImageEnhance
import gc
import pandas as pd
from PIL import Image
import cv2

import tensorflow as tf

save_directory = '/media/olle/Seagate Expansion Drive/DRD_master_thesis_olle_holmberg/kaggle/test_tfrecords_v2/'
label_directory = '/media/olle/Seagate Expansion Drive/DRD_master_thesis_olle_holmberg/kaggle/zip_files/'
data_directory = '/media/olle/Seagate Expansion Drive/DRD_master_thesis_olle_holmberg/kaggle/zip_files/test/'
#@profile

def scaleRadius(img,scale):
    x=img[int(img.shape[0]/2),:,:].sum(1)
    r=(x>x.mean()/10).sum()/2
    s=scale*1.0/r
    return cv2.resize(img,(0,0),fx=s, fy=s)

def data_list(data_dir, label_dir, num_examples, k, filenames):
    '''
    imports: pandas, os, numpy, PIL
    '''
    class_factors = np.array([ 1, 12,  4,38, 47])
    scale = 512
    width = 512
    height = 512
    # get labels csv into pandas df
    # below line assumes
    label_file_name = os.listdir(label_dir)[0]
    label_pd = pd.read_csv(label_dir + 'retinopathy_solution.csv', engine='python')
    label_pd = label_pd.loc[label_pd['Usage'] == "Private"][["image","level"]]    # initilize container list
    data = [[], [], [], [], []]
    # get filenames om images
    #filenames = label_pd['image'].values
    print(type(filenames))
    iter = 0
    # below loop retrieved the
    for im_number in range(k * num_examples, (k + 1) * num_examples):
        try:
            print(data_dir + filenames[im_number])
            path = data_dir + filenames[im_number]#+".jpeg"
            #try:
            image_path = os.path.join(path)
            a = cv2.imread(image_path)
            # scale img to a given radius
            a = scaleRadius(a, scale)
            # s u b t r a c t l o c a l mean c o l o r
            a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)
            # remove outer 10%
            b = np.zeros(a.shape)
            x = int(a.shape[1]/2)#.astype("int")
            y = int(a.shape[0]/2)#.astype("int")
            radius = int(scale * 0.9)#.astype("int")
            cv2.circle(b, (x, y), radius, (1, 1, 1), -1, 8, 0)
            im_procc = a * b + 128 * (1 - b)

            #standardize size of im_procc
            im_procc = cv2.resize(im_procc, (width, height)).astype(np.uint8)

            # convert to numpy formatat before appending to list
            im = im_procc.reshape(1, width, height, 3)
            name = filenames[im_number].replace(".jpeg", "")
            label = label_pd.loc[label_pd['image'] == name].iloc[0]['level']
            image_mean = np.mean(im)
            image_std = np.std(im)

            data[0].append(name)
            data[1].append(im)
            data[2].append(label)
            data[3].append(image_mean)
            data[4].append(image_std)

            iter += 1
            print(iter)
            if iter % 10 == 0:

                #print("{} images loaded".format(im_number))
                print("number of images successfully loaded {}".format(k))
            #gc.collect()
            if iter == 5:
                sdjajsdja
        except:
            print("except")
            continue
    return (data)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#@profile
def convert_to(train_images,train_labels,train_name,train_num_exmaples,train_images_mean,train_images_std, name):
    """Converts a dataset to tfrecords."""
    images = train_images
    labels = train_labels
    print("train im shape is {}".format(images.shape))
    image_means = train_images_mean
    image_stds = train_images_std
    num_examples = train_num_exmaples
    image_name = train_name
    print('number of examples this file is {}'.format(num_examples))
    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(save_directory, name + '.bin')
    print('Writing', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(num_examples):
            if index%100 == 0:
                gc.collect()
            #print(images.shape)
            image_raw = images[index].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(rows),
                        'width': _int64_feature(cols),
                        'depth': _int64_feature(depth),
                        'image_mean': _floats_feature(image_means[index]),
                        'image_std': _floats_feature(image_stds[index]),
                        'label': _int64_feature(int(labels[index])),
                        'image_raw': _bytes_feature(image_raw),
                        'image_name': _bytes_feature(train_name[index]),

                    }))
            writer.write(example.SerializeToString())
#@profile
def main():
    filenames = os.listdir(data_directory)
    num_examples = int(len(filenames) / 20)
    print("number of exmaples is {}".format(num_examples))
    for i in range(0, 20):

        data = data_list(data_directory, label_directory, num_examples, i, filenames)

        train_images = np.vstack(data[1])
        train_labels = np.asarray(data[2])
        train_name = data[0]
        train_num_exmaples = train_labels.shape[0]
        train_images_mean = np.asarray(data[3])
        train_images_std = np.asarray(data[4])
        print("convert new batch to tfrecords")
        # Convert to Examples and write the result to TFRecords.
        convert_to(train_images,train_labels,train_name,train_num_exmaples,train_images_mean,train_images_std, "data_batch_{}".format(i))
        del data
        del train_images
        gc.collect()

main()