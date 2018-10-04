from enum import Enum

import tensorflow as tf
import cv2


regularizer_conv = 0.004
regularizer_dsconv = 0.0004
batchnorm_fused = True # 如果“True”，如果可能的话，使用一个更快的融合实现。 如果“False”，请使用系统建议的实现。
activation_fn = tf.nn.relu
