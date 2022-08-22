# -*- coding: UTF-8 -*-
"""
construct the classification network of PGVNet (Point & Group-View Net)
__author__ = Cush
Created on August 2020
"""

import tensorflow as tf
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from models.transform_nets import input_transform_net
from models import vgg16

slim = tf.contrib.slim


class PGVNet(object):
    def __init__(self, num_classes, num_group, is_training, vgg16_npy_path, bn_decay=None):
        super(PGVNet, self).__init__()
        self.num_classes = num_classes
        self.num_group = num_group
        self.is_training = is_training
        self.bn_decay = bn_decay

        if vgg16_npy_path is not None:
            self.vgg = self.init_views_branch(vgg16_npy_path)

    @staticmethod
    def init_views_branch(vgg16_npy_path_):
        return vgg16.Vgg16(vgg16_npy_path_)

    def point_branch(self, point_cloud):
        """
        input is BxNx3,
        output (batch_size, Num_point, 1, 64) ---> (N, 64)
        """
        k = 10
        adj_matrix = tf_util.pairwise_distance(point_cloud)
        nn_idx = tf_util.knn(adj_matrix, k=k)
        edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)
        atedge_feature = tf_util.get_attention_edge_feature_layer(edge_feature, 64, activation=tf.nn.elu,
                                                                  is_training=self.is_training,
                                                                  bn_decay=self.bn_decay,
                                                                  layer='layer0')

        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(atedge_feature, self.is_training, self.bn_decay, K=3)

        point_cloud_transformed = tf.matmul(point_cloud, transform)
        adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
        nn_idx = tf_util.knn(adj_matrix, k=k)
        edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)
        atedge_feature = tf_util.get_attention_edge_feature_layer(edge_feature, 64, activation=tf.nn.elu,
                                                                  is_training=self.is_training,
                                                                  bn_decay=self.bn_decay,
                                                                  layer='layer1')
        net1 = atedge_feature
        adj_matrix = tf_util.pairwise_distance(atedge_feature)
        nn_idx = tf_util.knn(adj_matrix, k=k)
        edge_feature = tf_util.get_edge_feature(atedge_feature, nn_idx=nn_idx, k=k)
        atedge_feature = tf_util.get_attention_edge_feature_layer(edge_feature, 64, activation=tf.nn.elu,
                                                                  is_training=self.is_training,
                                                                  bn_decay=self.bn_decay,
                                                                  layer='layer2')
        net2 = atedge_feature
        net = tf_util.conv2d(tf.concat([net1, net2], axis=-1), 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training,
                             scope='pgv_agg', bn_decay=self.bn_decay)
        return net

    @staticmethod
    def cal_scores(scores):
        n = len(scores)
        s = tf.ceil(scores[0] * n)
        for idx, score in enumerate(scores):
            if idx == 0:
                continue
            s += tf.ceil(score * n)
        s /= n
        return s

    @staticmethod
    def group_fusion(view_group, weight_group):
        shape_des = map(lambda a, b: a*b, view_group, weight_group)
        shape_des = sum(shape_des)/sum(weight_group)
        return shape_des

    def group_pooling(self, final_views, views_score, num_group):
        interval = 1.0/num_group

        def onebatch_grouping(onebatch_views, onebatch_scores):
            viewgroup_onebatch = [[] for i in range(num_group)]
            scoregroup_onebatch = [[] for i in range(num_group)]

            for i in range(num_group):
                left = i * interval
                right = (i + 1) * interval
                shape_batch_scores = onebatch_scores.get_shape()[0]
                for j in range(shape_batch_scores):
                    score = onebatch_scores[j]
                    if (tf.greater_equal(score, left) & tf.less(score, right)) is not False:
                        viewgroup_onebatch[i].append(onebatch_views[j])
                        scoregroup_onebatch[i].append(score)

            view_group = [sum(views) / len(views) for views in viewgroup_onebatch if len(views) > 0]
            weight_group = [self.cal_scores(scores) for scores in scoregroup_onebatch if len(scores) > 0]
            onebatch_shape_des = self.group_fusion(view_group, weight_group)
            return onebatch_shape_des
        shape_descriptors = []

        batch_size = final_views.get_shape()[0]
        for i_ in range(batch_size):
            onebatch_views = final_views[i_]
            onebatch_scores = views_score[i_]
            shape_descriptors.append(onebatch_grouping(onebatch_views, onebatch_scores))
        shape_descriptor = tf.stack(shape_descriptors, 0)
        return shape_descriptor

    def views_branch(self, multi_view_image):
        batch_size, num_views, image_size_H, image_size_W, channel = multi_view_image.shape[0], multi_view_image.shape[1], \
                                                                     multi_view_image.shape[2], multi_view_image.shape[3], multi_view_image.shape[4]

        views = tf.reshape(multi_view_image, [batch_size*num_views, image_size_H, image_size_W, channel])

        self.vgg.build(views)

        raw_views = self.vgg.pool3
        raw_views = tf.keras.layers.GlobalAveragePooling2D()(raw_views)
        raw_views = tf.keras.layers.Dense(1, name='gv_raw')(raw_views)
        views_score = tf.nn.sigmoid(tf.log(tf.abs(raw_views)))
        views_score = tf.reshape(views_score, [batch_size, num_views])

        final_view_descriptors = tf.keras.layers.GlobalAveragePooling2D(name='gv_gap')(self.vgg.pool5)
        final_view_descriptors = tf.reshape(final_view_descriptors, [batch_size, num_views, -1])

        shape_descriptor = self.group_pooling(final_view_descriptors, views_score, self.num_group)
        return shape_descriptor

    @staticmethod
    def embedding_net(views_branch_shape_descriptors):
        """
        input:views_branch_shape_descriptors
        return: embedding view feature  1x1024
        """
        embedded_view_feature = tf.keras.layers.Dense(1024, name='embedding_net')(views_branch_shape_descriptors)
        return embedded_view_feature

    @staticmethod
    def attention_fusion_block(name, point_branch_feature, embedded_view_feature, is_training, bn_decay, num_node=64, k=5):
        """
        input:
            point_branch_feature: N x 64
            embedded_view_feature: 1 x 1024
        output:
            N x 64 features
        """
        N = point_branch_feature.get_shape()[1].value
        embedded_view_feature_repeated = tf.tile(embedded_view_feature, [1, N, 1, 1])
        contact_feature = tf.concat([embedded_view_feature_repeated, point_branch_feature], axis=-1)

        contact_feature = tf_util.conv2d(contact_feature, num_node, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=True, is_training=is_training,
                                         scope=name+'_afb_1', bn_decay=bn_decay)

        contact_feature = tf.reduce_max(contact_feature, axis=-2, keep_dims=True)

        # normalization
        soft_attention_mask = tf.nn.sigmoid(tf.log(tf.clip_by_value(abs(contact_feature), 1e-8, 1.0)))

        adj_matrix = tf_util.pairwise_distance(point_branch_feature)
        nn_idx = tf_util.knn(adj_matrix, k=k)
        edge_feature = tf_util.get_edge_feature(point_branch_feature, nn_idx=nn_idx, k=k)
        point_branch_feature_conv = tf_util.conv2d(edge_feature, num_node, [1, 1],
                                                   padding='VALID', stride=[1, 1],
                                                   bn=True, is_training=is_training,
                                                   scope=name+'_afb_2', bn_decay=bn_decay)

        point_branch_feature_conv = tf.reduce_max(point_branch_feature_conv, axis=-2, keep_dims=True)
        refined_feature = tf.multiply(point_branch_feature_conv, soft_attention_mask)

        output_feature = tf.add(refined_feature, point_branch_feature_conv)
        return output_feature

    def embedding_attention_fusion(self, point_branch_feature, views_branch_shape_descriptors):
        """
        input:
        point_branch_feature
        views_branch_shape_descriptors
        return: embedded_view_feature
        """
        embedded_view_feature = self.embedding_net(views_branch_shape_descriptors)
        embedded_view_feature = tf.expand_dims(embedded_view_feature, axis=-2)
        embedded_view_feature = tf.expand_dims(embedded_view_feature, axis=-2)

        name_1 = 'fir'
        afb1 = self.attention_fusion_block(name_1, point_branch_feature,
                                           embedded_view_feature, self.is_training,
                                           self.bn_decay, Num_node=64, k=10)
        name_2 = 'sec'
        afb2 = self.attention_fusion_block(name_2, afb1,
                                           embedded_view_feature, self.is_training,
                                           self.bn_decay, Num_node=128, k=10)
        return embedded_view_feature, afb2

    def forward(self, point_cloud, multi_view_image):
        """
        input:
            point_cloud:--> (batch_size, Num_point, 3)
            multi_view_image:--> (batch_size, Num_view, height, width, 3)
        """
        batch_size = point_cloud.get_shape()[0].value

        point_branch_feature = self.point_branch(point_cloud[:, :, :3])
        views_branch_shape_descriptors = self.views_branch(multi_view_image)

        embedded_view_feature, attention_emdedding_feature = \
            self.Embedding_attention_fusion(point_branch_feature, views_branch_shape_descriptors)

        net0 = tf.concat([point_branch_feature, attention_emdedding_feature], axis=-1)
        net0 = tf_util.conv2d(net0, 1024, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=True, is_training=self.is_training,
                              scope='pgv_conv2d', bn_decay=self.bn_decay)
        net0 = tf.reduce_max(net0, axis=-3, keep_dims=True)

        fused_feature = tf.concat([embedded_view_feature, net0], axis=-1)
        net = tf.reshape(fused_feature, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=self.is_training,
                                      scope='pgv_fc1', bn_decay=self.bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5, is_training=self.is_training,
                              scope='pgv_dp1')
        net = tf_util.fully_connected(net, 256, bn=True, is_training=self.is_training,
                                      scope='pgv_fc2', bn_decay=self.bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5, is_training=self.is_training,
                              scope='pgv_dp2')
        net = tf_util.fully_connected(net, self.num_classes, activation_fn=None, scope='pgv_fc3')
        net = tf.nn.softmax(net)

        return net


def placeholder_inputs(batch_size, num_point, num_views, height, width):
    # point cloud placeholder
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3), name='pointclouds_pl')

    # multi-view placeholder
    multi_view_pl = tf.placeholder(tf.float32, shape=(batch_size, num_views, height, width, 3), name='multi_view_pl')
    labels_pl = tf.placeholder(tf.int32, shape=batch_size, name='labels_pl')
    return pointclouds_pl, multi_view_pl, labels_pl


def get_loss(pred, label, num_classes):
    """
    pred: B*NUM_CLASSES,
    label: B
    """
    labels = tf.one_hot(indices=label, depth=num_classes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    classify_loss = tf.reduce_mean(loss) + tf.add_n(tf.get_collection('losses'))

    return classify_loss


if __name__ == '__main__':
    pass
