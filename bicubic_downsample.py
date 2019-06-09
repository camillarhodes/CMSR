# TensorFlow Better Bicubic Downsample
# https://github.com/trevor-m/tensorflow-bicubic-downsample
import numpy as np
import tensorflow as tf

FILTER_SIZE = 4

def bicubic_kernel(x, a=-0.5):
  """https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic"""
  if abs(x) <= 1:
    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
  elif 1 < abs(x) and abs(x) < 2:
    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
  else:
    return 0

def build_filter(factor_h, factor_w):
  # size = factor_h*factor_w
  # TODO: optimize
  size = FILTER_SIZE
  k = np.zeros((size))
  for i in range(size):
    x = (1/size)*(i- np.floor(size/2) +0.5)
    k[i] = bicubic_kernel(x)
  k = k / np.sum(k)
  # make 2d
  k = np.outer(k, k.T)
  k = tf.constant(k, dtype=tf.float32, shape=(size, size, 1, 1))
  return tf.concat([k, k, k], axis=2)

def apply_bicubic_downsample(x, filter, factor_h, factor_w):
  # using padding calculations from https://www.tensorflow.org/api_guides/python/nn#Convolution
  # filter_height = factor_h
  # filter_width = factor_w
  filter_height = FILTER_SIZE
  filter_width = FILTER_SIZE
  # strides = factor
  pad_along_height = max(filter_height - factor_h, 0)
  pad_along_width = max(filter_width - factor_w, 0)
  # compute actual padding values for each side
  pad_top = pad_along_height // 2
  pad_bottom = pad_along_height - pad_top
  pad_left = pad_along_width // 2
  pad_right = pad_along_width - pad_left
  # apply mirror padding
  x = tf.pad(x, [[0,0], [pad_top,pad_bottom], [pad_left,pad_right], [0,0]], mode='REFLECT')
  # downsampling performed by strided conv
  x = tf.nn.depthwise_conv2d(x, filter=filter, strides=[1,factor_h,factor_w,1], padding='VALID')
  return x
