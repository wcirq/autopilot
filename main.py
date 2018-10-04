import tensorflow as tf

from src.network import MobilenetNetworkThin

input_shape = (720, 1280, 3) # (高度， 宽度， 通道)
net = MobilenetNetworkThin(input_shape)
