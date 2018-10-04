import os
import tensorflow as tf
import time

from src.network_base import BaseNetwork


class MobilenetNetworkThin(BaseNetwork):
    def __init__(self, input_shape, keep_prob=1.0, trainable=True, conv_width=1.0, conv_width2=None):
        placeholder_input = tf.placeholder("float", input_shape)
        inputs = {'image': placeholder_input}
        self.num_refine = 5
        self.keep_prob = keep_prob
        self.conv_width = conv_width
        self.conv_width2 = conv_width2 if conv_width2 else conv_width
        BaseNetwork.__init__(self, inputs, trainable)

    def setup(self):
        min_depth = 8
        depth = lambda d: max(int(d * self.conv_width), min_depth)
        depth2 = lambda d: max(int(d * self.conv_width2), min_depth)

        with tf.variable_scope(None, 'feature_extraction'):
            (self.feed('image')
                .normalize_input(name="normalize")
                .convb(3, 3, depth(32), 4, name='Conv2d_0')
                .separable_conv(1, 1, depth(16), 1, name='Conv2d_1_1')  # 卷积核高,卷积核宽,输出map数，步长
                .separable_conv(3, 3, depth(16), 1, name='Conv2d_1_2')  # 卷积核高,卷积核宽,输出map数，步长
                .separable_conv(1, 1, depth(64), 1, name='Conv2d_1_3')  # 卷积核高,卷积核宽,输出map数，步长

                .separable_conv(1, 1, depth(32), 1, name='Conv2d_2_1')
                .separable_conv(3, 3, depth(32), 1, name='Conv2d_2_2')
                .separable_conv(1, 1, depth(128), 2, name='Conv2d_2_3')

                .separable_conv(1, 1, depth(32), 1, name='Conv2d_3_1')
                .separable_conv(3, 3, depth(32), 1, name='Conv2d_3_2')
                .separable_conv(1, 1, depth(128), 1, name='Conv2d_3_3')

                .separable_conv(1, 1, depth(64), 1, name='Conv2d_4_1')
                .separable_conv(3, 3, depth(64), 1, name='Conv2d_4_2')
                .separable_conv(1, 1, depth(256), 2, name='Conv2d_4_3')

                .separable_conv(1, 1, depth(64), 1, name='Conv2d_5_1')
                .separable_conv(3, 3, depth(64), 1, name='Conv2d_5_2')
                .separable_conv(1, 1, depth(256), 1, name='Conv2d_5_3')

                .separable_conv(1, 1, depth(128), 1, name='Conv2d_6_1')
                .separable_conv(3, 3, depth(128), 1, name='Conv2d_6_2')
                .separable_conv(1, 1, depth(512), 1, name='Conv2d_6_3')

                .separable_conv(1, 1, depth(128), 1, name='Conv2d_7_1')
                .separable_conv(3, 3, depth(128), 1, name='Conv2d_7_2')
                .separable_conv(1, 1, depth(512), 1, name='Conv2d_7_3')

                .separable_conv(1, 1, depth(128), 1, name='Conv2d_8_1')
                .separable_conv(3, 3, depth(128), 1, name='Conv2d_8_2')
                .separable_conv(1, 1, depth(512), 1, name='Conv2d_8_3')

                .separable_conv(1, 1, depth(128), 1, name='Conv2d_9_1')
                .separable_conv(3, 3, depth(128), 1, name='Conv2d_9_2')
                .separable_conv(1, 1, depth(512), 1, name='Conv2d_9_3')

                .separable_conv(1, 1, depth(128), 1, name='Conv2d_10_1')
                .separable_conv(3, 3, depth(128), 1, name='Conv2d_10_2')
                .separable_conv(1, 1, depth(512), 1, name='Conv2d_10_3')

                .separable_conv(1, 1, depth(128), 1, name='Conv2d_11_1')
                .separable_conv(3, 3, depth(128), 1, name='Conv2d_11_2')
                .separable_conv(1, 1, depth(512), 1, name='Conv2d_11_3')
                )

        with tf.variable_scope(None, 'feature_merge'):
            (self.feed('Conv2d_3_3')
                .max_pool(2, 2, 2, 2, name='Conv2d_3_pool'))

            feature_lv = 'feat_concat'
            (self.feed('Conv2d_3_pool', 'Conv2d_5_3', 'Conv2d_11_3')
                .concat(3, name=feature_lv))






    def loss_l1_l2(self):
        """
        返回各阶段的最后一层
        :return:
        """
        l1s = []
        l2s = []
        for layer_name in sorted(self.layers.keys()):
            if '_L1_5' in layer_name:
                l1s.append(self.layers[layer_name])
            if '_L2_5' in layer_name:
                l2s.append(self.layers[layer_name])

        return l1s, l2s

    def last_layer(self):
        """
        返回最后一阶段的两个MAP
        :return:
        """
        heat = self.get_output('MConv_Stage{0}_L1_5'.format(self.num_refine + 1))
        vect = self.get_output('MConv_Stage{0}_L2_5'.format(self.num_refine + 1))
        return heat, vect
        # return self.get_output("output_layer")

    def restorable_variables(self):
        vs = {v.op.name: v for v in tf.global_variables() if
              'MobilenetV1/Conv2d' in v.op.name and
              # 'global_step' not in v.op.name and
              # 'beta1_power' not in v.op.name and 'beta2_power' not in v.op.name and
              'RMSProp' not in v.op.name and 'Momentum' not in v.op.name and
              'Ada' not in v.op.name and 'Adam' not in v.op.name
              }
        return vs


if __name__ == '__main__':
    pass
