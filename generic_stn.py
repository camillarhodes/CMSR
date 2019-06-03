import numpy as np
from stn.transformer import bilinear_sampler

def generic_spatial_transformer_network(input_fmap, batch_grids):
    # batch_grids = tf.Print(batch_grids, [batch_grids[0,:,65:70,65:70]],summarize=-1)


    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)

    return out_fmap


def generic_grid_generator(height, width, num_batch):

    # create normalized 2D grid
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x_t, y_t = np.meshgrid(x, y)

    sampling_grid = np.stack([x_t, y_t])

    # repeat grid num_batch times
    # TODO: validate this works with num_batch > 1
    sampling_grid = np.expand_dims(sampling_grid, axis=0)
    sampling_grid = np.tile(sampling_grid, np.stack([num_batch, 1, 1, 1]))

    # reshape to (num_batch, H, W, 2)
    batch_grids = np.reshape(sampling_grid, [num_batch, 2, height, width])

    return batch_grids
