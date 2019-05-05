import tensorflow as tf
from stn.transformer import bilinear_sampler

def generic_spatial_transformer_network(input_fmap, batch_grids):
    batch_grids = tf.Print(batch_grids, [batch_grids[0,:,60:70,60:70]],summarize=-1)


    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)

    return out_fmap


def generic_grid_generator(height, width, num_batch):

    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    sampling_grid = tf.stack([x_t, y_t])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    #sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(sampling_grid, [num_batch, 2, height, width])

    return batch_grids
