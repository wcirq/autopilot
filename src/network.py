import os
import tensorflow as tf
import time

from src.network_base import BaseNetwork


class MobilenetNetworkThin(BaseNetwork):
    def __init__(self, input_shape, keep_prob=1.0, trainable=True, conv_width=1.0, conv_width2=None):
        placeholder_input = tf.placeholder("float", input_shape)
        inputs = {'image': placeholder_input}
        self.num_refine = 3
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

        with tf.variable_scope(None, 'feature_refining'):
            with tf.variable_scope(None, 'stage1'):
                (self.feed(feature_lv)
                    .separable_conv(1, 1, depth(128), 1, name='stage1_branch_1_1')
                    .separable_conv(3, 3, depth(128), 1, name='stage1_branch_1_2')
                    .separable_conv(1, 1, depth(512), 1, name='stage1_branch_1')
                )
                (self.feed(feature_lv)
                    .separable_conv(1, 1, depth(512), 1, name='stage1_branch_2')
                )
                (self.feed(feature_lv)
                    .separable_conv(1, 1, depth(128), 1, name='stage1_branch_3_1')
                    .separable_conv(3, 3, depth(128), 1, name='stage1_branch_3_2')
                    .separable_conv(1, 1, depth(512), 1, name='stage1_branch_3')
                )
                (self.feed('stage1_branch_1', 'stage1_branch_2', 'stage1_branch_3', 'Conv2d_3_pool')
                    .concat(3, name='feat_ref_concat_0'))

            for index in range(self.num_refine):
                stage = 'stage{}_'.format(index+2)
                with tf.variable_scope(None, stage):
                    (self.feed('feat_ref_concat_{}'.format(index))
                        .separable_conv(1, 1, depth(128), 1, name=stage+'branch_1_1')
                        .separable_conv(3, 3, depth(128), 1, name=stage+'branch_1_2')
                        .separable_conv(1, 1, depth(512), 1, name=stage+'branch_1')
                        )
                    (self.feed('feat_ref_concat_{}'.format(index))
                        .separable_conv(1, 1, depth(512), 1, name=stage+'branch_2')
                        )
                    (self.feed('feat_ref_concat_{}'.format(index))
                        .separable_conv(1, 1, depth(128), 1, name=stage+'branch_3_1')
                        .separable_conv(3, 3, depth(128), 1, name=stage+'branch_3_2')
                        .separable_conv(1, 1, depth(512), 1, name=stage+'branch_3')
                        )
                    (self.feed(stage+'branch_1', stage+'branch_2', stage+'branch_3', 'Conv2d_3_pool')
                        .concat(3, name='feat_ref_concat_{}'.format(index+1)))

        with tf.variable_scope(None, 'export'):
            name = 'drivable'
            with tf.variable_scope(None, name):
                (self.feed('feat_ref_concat_{}'.format(self.num_refine))
                    .separable_conv(1, 1, depth(128), 1, name=name+'_Conv2d_1_1')
                    .separable_conv(3, 3, depth(128), 1, name=name+'_Conv2d_1_2')
                    .separable_conv(1, 1, depth(512), 1, name=name+'_Conv2d_1_3')

                    .separable_conv(1, 1, depth(128), 1, name=name+'_Conv2d_2_1')
                    .separable_conv(3, 3, depth(128), 1, name=name+'_Conv2d_2_2')
                    .separable_conv(1, 1, depth(512), 1, name=name+'_Conv2d_2_3')

                    .separable_conv(1, 1, depth(64), 1, name=name+'_Conv2d_3_1')
                    .separable_conv(3, 3, depth(64), 1, name=name+'_Conv2d_3_2')
                    .separable_conv(1, 1, depth(256), 1, name=name+'_Conv2d_3_3')

                    .separable_conv(1, 1, depth(64), 1, name=name+'_Conv2d_4_1')
                    .separable_conv(3, 3, depth(64), 1, name=name+'_Conv2d_4_2')
                    .separable_conv(1, 1, depth(256), 1, name=name+'_Conv2d_4_3')

                    .separable_conv(1, 1, depth(32), 1, name=name+'_Conv2d_5_1')
                    .separable_conv(3, 3, depth(32), 1, name=name+'_Conv2d_5_2')
                    .separable_conv(1, 1, depth(2), 1, name=name+'_Conv2d_5_3')
                )
            name = 'box2d'
            with tf.variable_scope(None, name):
                (self.feed('feat_ref_concat_{}'.format(self.num_refine))
                    .separable_conv(1, 1, depth(128), 1, name=name + '_Conv2d_1_1')
                    .separable_conv(3, 3, depth(128), 1, name=name + '_Conv2d_1_2')
                    .separable_conv(1, 1, depth(512), 1, name=name + '_Conv2d_1_3')

                    .separable_conv(1, 1, depth(128), 1, name=name + '_Conv2d_2_1')
                    .separable_conv(3, 3, depth(128), 1, name=name + '_Conv2d_2_2')
                    .separable_conv(1, 1, depth(512), 1, name=name + '_Conv2d_2_3')

                    .separable_conv(1, 1, depth(64), 1, name=name + '_Conv2d_3_1')
                    .separable_conv(3, 3, depth(64), 1, name=name + '_Conv2d_3_2')
                    .separable_conv(1, 1, depth(256), 1, name=name + '_Conv2d_3_3')

                    .separable_conv(1, 1, depth(64), 1, name=name + '_Conv2d_4_1')
                    .separable_conv(3, 3, depth(64), 1, name=name + '_Conv2d_4_2')
                    .separable_conv(1, 1, depth(256), 1, name=name + '_Conv2d_4_3')

                    .separable_conv(1, 1, depth(32), 1, name=name + '_Conv2d_5_1')
                    .separable_conv(3, 3, depth(32), 1, name=name + '_Conv2d_5_2')
                    .separable_conv(1, 1, depth(30), 1, name=name + '_Conv2d_5_3')
                    )
                print()



if __name__ == '__main__':
    pass
