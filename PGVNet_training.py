# -*- coding: UTF-8 -*-
"""
construct the train graph of PGVNet (Point & Group-View Net)
input data : point cloud h5 file and png tfrecord file
__author__ = Cush
Created on May 2020
"""

import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import time
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from train_utils import *
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument('--model', default='PGVNet', help='Model name')
parser.add_argument('--log_dir', default='./log/PGVNet', help='Log dir')
parser.add_argument('--max_epoch', type=int, default=30, help='Epoch to run')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate')
parser.add_argument('--optimizer', default='momentum', help='adam or momentum')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay')
parser.add_argument('--pgv_iter_train', default=True,
                    help='Where the pretrained pointcloud branch reside.')
parser.add_argument('--point_checkpoint', default='/log/log_DAGNet/',
                    help='Where the pretrained pointcloud branch reside.')
parser.add_argument('--vgg16_npy_path', default='./log/log_vgg_imagenet/vgg16.npy',
                    help='Where the pretrained vgg model parameter reside.')
parser.add_argument('--pc_training_data_dir', default='./data/training_data/train_pc_files.txt',
                    help='Where the point cloud training dataset reside.')
parser.add_argument('--pc_test_data_dir', default='./data/training_data/test_pc_files.txt',
                    help='Where the point cloud test dataset reside.')
parser.add_argument('--num_object_each_h5file', type=int, default=2048, help='width.')
parser.add_argument('--num_point', type=int, default=2048, help='Number point of each object.')
parser.add_argument('--im_training_data_dir', default='./data/training_data/train_views_files.txt',
                    help='Where the multi-view training dataset reside.')
parser.add_argument('--im_test_data_dir', default='./data/training_data/test_views_files.txt',
                    help='Where the multi-view test dataset reside.')
parser.add_argument('--num_views', type=int, default=8, help='Number of views.')
parser.add_argument('--num_group', type=int, default=10, help='number of group. ')
parser.add_argument('--im_height', type=int, default=224, help='height.')
parser.add_argument('--im_width', type=int, default=224, help='width. ')
parser.add_argument('--labels', default='car, tree, pole, small_object, other',
                    help='Category name of object.')

FLAGS = parser.parse_args()

TRAIN_FILES = getDataFiles(FLAGS.pc_training_data_dir)
TEST_FILES = getDataFiles(FLAGS.pc_test_data_dir)

TRAIN_TFrecord_FILES = getDataFiles(FLAGS.im_training_data_dir)
TEST_TFrecord_FILES = getDataFiles(FLAGS.im_test_data_dir)

labels = FLAGS.labels.split(',')
NUM_CLASSES = len(labels)

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(FLAGS.decay_step)
BN_DECAY_CLIP = 0.99

MODEL = importlib.import_module(FLAGS.model)
if not os.path.exists(FLAGS.log_dir): os.mkdir(FLAGS.log_dir)
LOG_FOUT = open(os.path.join(FLAGS.log_dir, 'log_train.txt'), 'w')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        FLAGS.learning_rate,
                        batch * FLAGS.batch_size,
                        FLAGS.decay_step,
                        FLAGS.decay_rate,
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*FLAGS.batch_size,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def load_one_tfrecord_png(tfrecord_file_name):
    data_files = tf.gfile.Glob(tfrecord_file_name)
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                           'img_width': tf.FixedLenFeature([], tf.int64),
                                           'img_height': tf.FixedLenFeature([], tf.int64),
                                           })
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    height = tf.cast(features['img_height'], tf.int32)
    width = tf.cast(features['img_width'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    channel = 3
    image = tf.reshape(image, [height, width, channel])
    return image, label


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointclouds_pl, multi_view_pl, ground_truth_pl = MODEL.placeholder_inputs(
                FLAGS.batch_size, FLAGS.num_point, FLAGS.num_views, FLAGS.im_height, FLAGS.im_width)
            is_training_pl = tf.placeholder(tf.bool, name='is_training')

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)

            pgvnet = MODEL.PGVNet(NUM_CLASSES, FLAGS.num_group, is_training_pl, FLAGS.vgg16_npy_path, bn_decay)
            pred = pgvnet.forward(pointclouds_pl, multi_view_pl)
            loss = MODEL.get_loss(pred, ground_truth_pl, NUM_CLASSES)

            learning_rate = get_learning_rate(batch)
            if FLAGS.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum)
            elif FLAGS.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            train_op = optimizer.minimize(loss, global_step=batch)

            fir_afb_1_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fir_afb_1')
            fir_afb_2_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fir_afb_2')
            sec_afb_1_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='sec_afb_1')
            sec_afb_2_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='sec_afb_2')
            pgv_conv2d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pgv_conv2d')
            pgv_fc1_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pgv_fc1')
            pgv_fc2_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pgv_fc2')
            pgv_fc3_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pgv_fc3')

            embedding_net_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='embedding_net')

            train_op_1 = optimizer.minimize(loss, global_step=batch,
                                            var_list=fir_afb_1_var + fir_afb_2_var + sec_afb_1_var + sec_afb_2_var
                                            + pgv_conv2d_var + pgv_fc1_var + pgv_fc2_var + pgv_fc3_var
                                            + embedding_net_var)

            saver = tf.train.Saver()

            # read tfrecord files
            train_image_0, label_0 = load_one_tfrecord_png(TRAIN_TFrecord_FILES[0])
            train_image_1, label_1 = load_one_tfrecord_png(TRAIN_TFrecord_FILES[1])
            test_image_0, label_test_0 = load_one_tfrecord_png(TEST_TFrecord_FILES[0])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True

        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            if FLAGS.point_checkpoint and os.path.isdir(FLAGS.point_checkpoint):
                variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=['layer0edge_feature', 'layer0edge_attention',
                                                                                              'transform_net1', 'layer1edge_feature',
                                                                                              'layer1edge_attention', 'layer2edge_feature',
                                                                                              'layer2edge_attention'])
                saver_restore = tf.train.Saver(variables_to_restore)
                saver_restore.restore(sess, tf.train.latest_checkpoint(FLAGS.point_checkpoint))

            ops = {'pointclouds_pl': pointclouds_pl,
                   'multiview_pl': multi_view_pl,
                   'labels_pl': ground_truth_pl,
                   'is_training_pl': is_training_pl,
                   'learning_rate': learning_rate,
                   'pred': pred,
                   'loss': loss,
                   'train_image_0': train_image_0,
                   'train_image_1': train_image_1,
                   'test_image_0': test_image_0}

            max_val_acc = 0.0
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for epoch in range(FLAGS.max_epoch):
                if FLAGS.pgv_iter_train:
                    if epoch < 10:
                        train_one_epoch(sess, ops, train_op_1)
                    else:
                        train_one_epoch(sess, ops, train_op)
                val_accuracy = eval_one_epoch(sess, ops)

                if max_val_acc < val_accuracy:
                    max_val_acc = val_accuracy
                    save_path = saver.save(sess, os.path.join(FLAGS.log_dir, "pgvnet_model.ckpt"))
                    log_string("Model saved in file: %s" % save_path)

            coord.request_stop()
            coord.join(threads)


def train_one_epoch(sess, ops, train_op):
    is_training = True
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    for fn in range(len(TRAIN_FILES)):
        current_data, current_label = loadDataFile(TRAIN_FILES[train_file_idxs[fn]])

        current_train_image_name = 'train_image_' + str(train_file_idxs[fn])
        num_of_current_train_views = FLAGS.num_views*(current_data.shape[0])
        current_im = []
        for _ in tqdm(range(num_of_current_train_views)):
            current_im_temp = sess.run(ops[current_train_image_name])
            current_im.append(current_im_temp)
        current_im = np.array(current_im)

        file_size = current_data.shape[0]
        num_batches = file_size // FLAGS.batch_size

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * FLAGS.batch_size
            end_idx = (batch_idx+1) * FLAGS.batch_size

            im_start_idx = batch_idx * FLAGS.batch_size * FLAGS.num_views
            im_end_idx = (batch_idx + 1) * FLAGS.batch_size * FLAGS.num_views

            current_batch_train_data = current_data[start_idx:end_idx, :, :]
            current_batch_data_label = current_label[start_idx:end_idx]

            current_batch_train_im = current_im[im_start_idx:im_end_idx, :, :, :]
            current_train_multi_views = current_batch_train_im.reshape(FLAGS.batch_size, FLAGS.num_views,
                                                                       FLAGS.im_width, FLAGS.im_height, 3)

            feed_dict = {ops['pointclouds_pl']: current_batch_train_data,
                         ops['multiview_pl']: current_train_multi_views,
                         ops['labels_pl']: current_batch_data_label,
                         ops['is_training_pl']: is_training}

            lr, pred_value, loss_value, _ = sess.run([ops['learning_rate'],
                                                      ops['pred'],
                                                      ops['loss'],
                                                      train_op], feed_dict=feed_dict)

            pred_val = np.argmax(pred_value, 1)
            correct = np.sum(pred_val == current_batch_data_label)
            total_correct += correct
            total_seen += FLAGS.batch_size
            loss_sum += loss_value

            mean_loss = loss_value/FLAGS.batch_size
            acc = correct/FLAGS.batch_size

            if batch_idx % 10 == 0:
                message = 'Tf{:02d} Step {:08d} lr={:5.3f} L_out={:5.3f} Acc={:4.2f}'
                log_string(message.format(fn, batch_idx, lr, mean_loss, acc))      
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))


def eval_one_epoch(sess, ops):
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    true_label = []
    pre_label = []

    for fn in range(len(TEST_FILES)):
        current_data, current_label = loadDataFile(TEST_FILES[fn])

        current_test_image_name = 'test_image_' + str(fn)
        num_of_current_test_views = FLAGS.num_views * (current_data.shape[0])
        current_im = []
        for _ in tqdm(range(num_of_current_test_views)):
            current_im_temp = sess.run(ops[current_test_image_name])
            current_im.append(current_im_temp)
        current_test_im = np.array(current_im)

        file_size = current_data.shape[0]
        num_batches = file_size // FLAGS.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * FLAGS.batch_size
            end_idx = (batch_idx+1) * FLAGS.batch_size

            im_start_idx = batch_idx * FLAGS.batch_size * FLAGS.num_views
            im_end_idx = (batch_idx + 1) * FLAGS.batch_size * FLAGS.num_views

            current_batch_test_data = current_data[start_idx:end_idx, :, :]
            current_batch_data_label = current_label[start_idx:end_idx]

            current_batch_test_im = current_test_im[im_start_idx:im_end_idx, :, :, :]
            current_test_multi_views = current_batch_test_im.reshape(FLAGS.batch_size, FLAGS.num_views,
                                                                     FLAGS.im_width, FLAGS.im_height, 3)

            feed_dict = {ops['pointclouds_pl']: current_batch_test_data,
                         ops['multiview_pl']: current_test_multi_views,
                         ops['labels_pl']: current_batch_data_label,
                         ops['is_training_pl']: is_training}

            loss_value, pred_value = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_value, 1)
            correct = np.sum(pred_val == current_batch_data_label)

            true_label.extend(current_batch_data_label)
            pre_label.extend(pred_val)

            total_correct += correct
            total_seen += FLAGS.batch_size
            loss_sum += loss_value
            for i in range(FLAGS.batch_size):
                l = int(current_batch_data_label[i])
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i] == l)

            mean_loss = loss_value / FLAGS.batch_size
            e_acc = correct / FLAGS.batch_size

            if batch_idx % 10 == 0:
                message = 'Testfile{:02d} Step {:08d} L_out={:5.3f} eval Acc={:4.2f}'
                log_string(message.format(fn, batch_idx, mean_loss, e_acc))

    log_string('mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))
    log_string('avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class, dtype=np.float))))
    return total_correct / float(total_seen)


if __name__ == '__main__':
    start = time.time()
    train()
    duration = (time.time() - start)
    log_string("running time:{:.2f} s".format(duration))
    LOG_FOUT.close()
