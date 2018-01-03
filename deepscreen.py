import tensorflow as tf
import sonnet as snt
from tensorflow.contrib import slim
import numpy as np


class Net(snt.AbstractModule):
    def __init__(self,
                 num_classes,
                 dropout_keep_prob=0.8,
                 weight_decay=4e-5,
                 name='network'):
        super(Net, self).__init__(name=name)
        self._num_classes = num_classes
        self._dropout_keep_prob = dropout_keep_prob
        self._weight_decay = weight_decay

    @classmethod
    def truc_normal(cls, stddev):
        return tf.truncated_normal_initializer(stddev=stddev)

    @property
    def image_size(self):
        return 70

    @classmethod
    def arg_scope(cls, weight_decay, is_training):
        batch_norm_params = {
            'decay': 0.99,
            'epsilon': 0.001,
            'scale': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        }
        weights_regularizer = slim.l2_regularizer(weight_decay)
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=weights_regularizer):
                with slim.arg_scope([slim.conv2d],
                                    weights_initializer=cls.truc_normal(0.1),
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params=batch_norm_params):
                    with slim.arg_scope(
                            [slim.conv2d, slim.max_pool2d, slim.avg_pool2d, slim.fully_connected]) as arg_sc:
                        return arg_sc

    def _build(self, inputs, is_training):
        with slim.arg_scope(self.arg_scope(weight_decay=self._weight_decay, is_training=is_training)):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                net = slim.conv2d(inputs, 32, [3, 3], scope='conv_1a_3x3')
                net = slim.conv2d(net, 64, [3, 3], scope='conv_2a_3x3')
                net = slim.conv2d(net, 80, [1, 1], scope='conv_3a_1x1')
                net = slim.conv2d(net, 192, [3, 3], scope='conv_4a_3x3')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool_5a_3x3')

            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                with tf.variable_scope('mixed_5b'):
                    with tf.variable_scope('branch_0'):
                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='conv_0a_1x1')
                    with tf.variable_scope('branch_1'):
                        branch_1 = slim.conv2d(net, 48, [1, 1], scope='conv_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='conv_0b_5x5')
                    with tf.variable_scope('branch_2'):
                        branch_2 = slim.conv2d(net, 64, [1, 1], scope='conv_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='conv_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='conv_0c_3x3')
                    with tf.variable_scope('branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='avgpool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='conv_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
                with tf.variable_scope('mixed_5c'):
                    with tf.variable_scope('branch_0'):
                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='conv_0a_1x1')
                    with tf.variable_scope('branch_1'):
                        branch_1 = slim.conv2d(net, 48, [1, 1], scope='conv_0b_1x1')
                        branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='conv_0c_5x5')
                    with tf.variable_scope('branch_2'):
                        branch_2 = slim.conv2d(net, 64, [1, 1], scope='conv_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='conv_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='conv_0c_3x3')
                    with tf.variable_scope('branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='avgpool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='conv_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
                with tf.variable_scope('mixed_5d'):
                    with tf.variable_scope('branch_0'):
                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='conv_0a_1x1')
                    with tf.variable_scope('branch_1'):
                        branch_1 = slim.conv2d(net, 48, [1, 1], scope='conv_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='conv_0b_5x5')
                    with tf.variable_scope('branch_2'):
                        branch_2 = slim.conv2d(net, 64, [1, 1], scope='conv_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='conv_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='conv_0c_3x3')
                    with tf.variable_scope('branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='avgpool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='conv_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)

                with tf.variable_scope('mixed_6a'):
                    with tf.variable_scope('branch_0'):
                        branch_0 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID', scope='conv_0a_1x1')
                    with tf.variable_scope('branch_1'):
                        branch_1 = slim.conv2d(net, 64, [1, 1], scope='conv_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='conv_0b_3x3')
                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2, padding='VALID', scope='conv_1a_1x1')
                    with tf.variable_scope('branch_2'):
                        branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='maxpool_1a_3x3')
                    net = tf.concat([branch_0, branch_1, branch_2], axis=3)

                with tf.variable_scope('mixed_6b'):
                    with tf.variable_scope('branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='conv_0a_1x1')
                    with tf.variable_scope('branch_1'):
                        branch_1 = slim.conv2d(net, 128, [1, 1], scope='conv_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='conv_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='conv_0c_7x1')
                    with tf.variable_scope('branch_2'):
                        branch_2 = slim.conv2d(net, 128, [1, 1], scope='conv_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='conv_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='conv_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='conv_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='conv_0e_1x7')
                    with tf.variable_scope('branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='avgpool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='conv_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)

                with tf.variable_scope('mixed_6c'):
                    with tf.variable_scope('branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='conv_0a_1x1')
                    with tf.variable_scope('branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='conv_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='conv_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='conv_0c_7x1')
                    with tf.variable_scope('branch_2'):
                        branch_2 = slim.conv2d(net, 160, [1, 1], scope='conv_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='conv_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='conv_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='conv_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='conv_0e_1x7')
                    with tf.variable_scope('branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='avgpool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='conv_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)

                with tf.variable_scope('mixed_6d'):
                    with tf.variable_scope('branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='conv_0a_1x1')
                    with tf.variable_scope('branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='conv_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='conv_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='conv_0c_7x1')
                    with tf.variable_scope('branch_2'):
                        branch_2 = slim.conv2d(net, 160, [1, 1], scope='conv_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='conv_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='conv_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='conv_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='conv_0e_1x7')
                    with tf.variable_scope('branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='avgpool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='conv_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)

                with tf.variable_scope('mixed_6e'):
                    with tf.variable_scope('branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='conv_0a_1x1')
                    with tf.variable_scope('branch_1'):
                        branch_1 = slim.conv2d(net, 192, [1, 1], scope='conv_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='conv_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='conv_0c_7x1')
                    with tf.variable_scope('branch_2'):
                        branch_2 = slim.conv2d(net, 192, [1, 1], scope='conv_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='conv_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='conv_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='conv_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='conv_0e_1x7')
                    with tf.variable_scope('branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='avgpool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='conv_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)

                with tf.variable_scope('mixed_7a'):
                    with tf.variable_scope('branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='conv_0a_1x1')
                        branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2, padding='VALID', scope='conv_1a_3x3')
                    with tf.variable_scope('branch_1'):
                        branch_1 = slim.conv2d(net, 192, [1, 1], scope='conv_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='conv_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='conv_0c_7x1')
                        branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID', scope='conv_1a_3x3')
                    with tf.variable_scope('branch_2'):
                        branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='maxpool_1a_3x3')
                    net = tf.concat([branch_0, branch_1, branch_2], axis=3)

                with tf.variable_scope('mixed_7b'):
                    with tf.variable_scope('branch_0'):
                        branch_0 = slim.conv2d(net, 320, [1, 1], scope='conv_0a_1x1')
                    with tf.variable_scope('branch_1'):
                        branch_1 = slim.conv2d(net, 384, [1, 1], scope='conv_0a_1x1')
                        branch_1 = tf.concat([
                            slim.conv2d(branch_1, 384, [1, 3], scope='conv_0b_1x3'),
                            slim.conv2d(branch_1, 384, [3, 1], scope='conv_0b_3x1')
                        ], axis=3)

                    with tf.variable_scope('branch_2'):
                        branch_2 = slim.conv2d(net, 448, [1, 1], scope='conv_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='conv_0b_3x3')
                        branch_2 = tf.concat([
                            slim.conv2d(branch_2, 384, [1, 3], scope='conv_0c_1x3'),
                            slim.conv2d(branch_2, 384, [3, 1], scope='conv_0d_3x1')
                        ], axis=3)

                    with tf.variable_scope('branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='avgpool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='conv_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)

                with tf.variable_scope('mixed_7c'):
                    with tf.variable_scope('branch_0'):
                        branch_0 = slim.conv2d(net, 320, [1, 1], scope='conv_0a_1x1')
                    with tf.variable_scope('branch_1'):
                        branch_1 = slim.conv2d(net, 384, [1, 1], scope='conv_0a_1x1')
                        branch_1 = tf.concat([
                            slim.conv2d(branch_1, 384, [1, 3], scope='conv_0b_1x3'),
                            slim.conv2d(branch_1, 384, [3, 1], scope='conv_0b_3x1')
                        ], axis=3)

                    with tf.variable_scope('branch_2'):
                        branch_2 = slim.conv2d(net, 448, [1, 1], scope='conv_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='conv_0b_3x3')
                        branch_2 = tf.concat([
                            slim.conv2d(branch_2, 384, [1, 3], scope='conv_0c_1x3'),
                            slim.conv2d(branch_2, 384, [3, 1], scope='conv_0d_3x1')
                        ], axis=3)

                    with tf.variable_scope('branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='avgpool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='conv_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)

                with tf.variable_scope('logits'):
                    net = slim.avg_pool2d(net, [8, 8], padding='VALID', scope='avgpool_1a_8x8')
                    net = slim.dropout(net, keep_prob=self._dropout_keep_prob, scope='dropout_1b')
                    logits = slim.conv2d(net, self._num_classes, [1, 1],
                                         activation_fn=None,
                                         normalizer_fn=None,
                                         scope='conv_1c_1x1')
                    logits = tf.squeeze(logits, [1, 2], name='squeezed')

        return logits


def create_init_op():
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    return init_op


def load_ckpt(sess, model_dir, variables_to_restore=None):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    model_path = ckpt.model_checkpoint_path
    if variables_to_restore is None:
        variables_to_restore = slim.get_variables_to_restore()
    restore_op, restore_fd = slim.assign_from_checkpoint(
        model_path, variables_to_restore)
    sess.run(restore_op, feed_dict=restore_fd)
    print(f'{model_path} loaded')


class DSModel:
    def __init__(self, sess, model_dir):
        self.sess = sess
        self.model = Net(num_classes=4)
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, 70, 70, 2])
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.logits = self.model(self.inputs, self.is_training)
        sess.run(create_init_op())
        load_ckpt(sess, model_dir)

    def predict(self, x):
        fd = {self.inputs: x, self.is_training: False}
        yp = self.sess.run(self.logits, feed_dict=fd)
        return np.argmax(yp, axis=-1)


# sess = tf.InteractiveSession()
# dsm = DSModel(sess, model_dir='model')
