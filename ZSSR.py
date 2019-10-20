import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import ipdb
import signal

from matplotlib.gridspec import GridSpec
from configs import Config
from utils import *
from generic_stn import generic_grid_generator, generic_spatial_transformer_network as generic_transformer
from ddtn.transformers.transformer_util import get_transformer_layer, get_transformer_dim, get_transformer_init_weights
from ddtn.transformers.setup_CPAB_transformer import setup_CPAB_transformer
from tf_unet.unet import create_conv_net


class ZSSR:
    # Basic current state variables initialization / declaration
    kernel = None
    learning_rate = None
    hr_father = None
    hr_guider = None
    hr_guider_deformed = None
    hr_guider_augmented = None
    lr_son = None
    sr = None
    sf = None
    gt_per_sf = None
    final_sr = None
    hr_fathers_sources = []

    # Output variables initialization / declaration
    reconstruct_output = None
    train_output = None
    output_shape = None

    # Counters and logs initialization
    iter = 0
    base_sf = [1.0, 1.0]
    base_ind = 0
    sf_ind = 0
    mse = []
    psnr = []
    mse_rec = []
    psnr_rec = []
    interp_rec_mse = []
    interp_mse = []
    mse_steps = []
    loss = []
    loss_rec = []
    learning_rate_change_iter_nums = []
    fig = None

    # Network tensors (all tensors end with _t to distinguish)
    learning_rate_t = None
    lr_son_t = None
    hr_father_t = None
    hr_guider_t = None
    hr_guider_with_shape_t = None
    hr_guider_deformed_t = None
    hr_guider_augmented_t = None
    hr_guider_features_t = None

    # initial_grid = None
    filters_t = None
    filters_t_guider = None
    layers_t = None
    layers_t_guider = None
    net_output_t = None
    loss_t = None
    loss_rec_t = None
    theta_cpab_t = None
    theta_tps_t = None
    theta_affine_t = None
    train_op = None
    train_grid_op = None
    train_affine_op = None
    train_tps_op = None
    train_cpab_op = None
    init_op = None

    # Parameters related to plotting and graphics
    plots = None
    loss_plot_space = None
    lr_son_image_space = None
    hr_father_image_space = None
    out_image_space = None

    # Tensorflow graph default
    sess = None

    def __init__(self, input_img, conf=Config(), ground_truth=None, guiding_img=None, kernels=None):
        # Acquire meta parameters configuration from configuration class as a class variable
        self.conf = conf

        # Read input image (can be either a numpy array or a path to an image file)
        self.input = input_img if type(input_img) is not str else img.imread(input_img)

        # For evaluation purposes, ground-truth image can be supplied.
        self.gt = ground_truth if type(ground_truth) is not str else img.imread(ground_truth)

        # To improve learning, guiding image can be supplied
        self.gi = guiding_img if type(guiding_img) is not str else img.imread(guiding_img)
        # self.gi=None

        # Normalize images
        self.input, self.gt, self.gi = normalize_images(self.input, self.gt, self.gi)

        # Preprocess the kernels. (see function to see what in includes).
        self.kernels = preprocess_kernels(kernels, conf)

        # Prepare TF default computational graph
        self.model = tf.Graph()

        # Build network computational graph
        self.build_network(conf)

        # Initialize network weights and meta parameters
        self.init_sess(init_weights=True)

        # The first hr father source is the input (source goes through augmentation to become a father)
        # Later on, if we use gradual sr increments, results for intermediate scales will be added as sources.
        self.hr_fathers_sources = [self.input]

        # We keep the input file name to save the output with a similar name. If array was given rather than path
        # then we use default provided by the configs
        self.file_name = input_img if type(input_img) is str else conf.name

    def run(self):
        # set breakpoint on C-c
        def debug_signal_handler(signal, frame, self=self):
            ipdb.set_trace()

        signal.signal(signal.SIGINT, debug_signal_handler)

        # Run gradually on all scale factors (if only one jump then this loop only happens once)
        for self.sf_ind, (sf, self.kernel) in enumerate(zip(self.conf.scale_factors, self.kernels)):

            print('** Start training for sf=', sf, ' **')

            # Relative_sf (used when base change is enabled. this is when input is the output of some previous scale)
            if np.isscalar(sf):
                sf = [sf, sf]
            self.sf = np.array(sf) / np.array(self.base_sf)
            self.output_shape = np.uint(np.ceil(np.array(self.input.shape[0:2]) * self.sf))

            # Downscale ground-truth to the intermediate sf size (for gradual SR).
            # This only happens if there exists ground-truth and sf is not the last one (or too close to it).
            # We use imresize with both scale and output-size, see comment in forward_backward_pass.
            # noinspection PyTypeChecker
            self.gt_per_sf, = normalize_images(np.clip(
                imresize(self.gt,
                         scale_factor=self.sf / self.conf.scale_factors[-1] if self.output_shape is None else None,
                         output_shape=self.output_shape,
                         kernel=self.conf.downscale_gt_method
                         ), 0, 1
            )) if (
                self.gt is not None and
                self.sf is not None and
                np.any(np.abs(self.sf - self.conf.scale_factors[-1]) > 0.01)
            ) else None

            # Initialize network weights and meta parameters
            self.init_sess(init_weights=self.conf.init_net_for_each_sf)

            # Train the network
            self.train()

            # Use augmented outputs and back projection to enhance result. Also save the result.
            post_processed_output = self.final_test()

            # Keep the results for the next scale factors SR to use as dataset
            self.hr_fathers_sources.append(post_processed_output)

            # In some cases, the current output becomes the new input. If indicated and if this is the right scale to
            # become the new base input. all of these conditions are checked inside the function.
            self.base_change()

            # Save the final output if indicated
            if self.conf.save_results:
                sf_str = ''.join('X%.2f' % s for s in self.conf.scale_factors[self.sf_ind])

                post_processed_output, = remove_n_channels_dim(post_processed_output)

                plt.imsave('%s/%s_zssr_%s.%s' %
                           (self.conf.result_path, os.path.basename(self.file_name)[:-4], sf_str, self.conf.img_ext),
                           post_processed_output, vmin=0, vmax=1, cmap=self.conf.cmap)
                # cv2.imwrite('%s/%s_zssr_%s.png' %
                #            (self.conf.result_path, os.path.basename(self.file_name)[:-4], sf_str),
                #            post_processed_output * 255)

            # verbose
            print('** Done training for sf=', sf, ' **')

        # Return the final post processed output.
        # noinspection PyUnboundLocalVariable
        return post_processed_output

    def build_network(self, meta):
        with self.model.as_default():

            # Learning rate tensor
            self.learning_rate_t = tf.placeholder(tf.float32, name='learning_rate')

            if self.gi is not None:

                # Most times we need to deform the guider. in final_test we do it
                # outside of the network so it's unneeded here
                should_deform_and_augment_guider = tf.placeholder_with_default(True, shape=(), name='should_deform_and_augment_guider')

                # Guider image
                self.hr_guider_t = tf.placeholder(tf.float32, name='hr_guider')

                # Guider image with shape, needed for TPS / affine
                # transformations
                self.hr_guider_with_shape_t = tf.placeholder(tf.float32,
                    np.expand_dims(add_n_channels_dim(self.gi)[0], 0).shape,
                    name='hr_guider_with_shape',
                )


            # Input image
            self.lr_son_t = tf.placeholder(tf.float32, name='lr_son')

            # Ground truth (supervision)
            self.hr_father_t = tf.placeholder(tf.float32, name='hr_father')

            if self.gi is not None:

                dim_tps = get_transformer_dim('TPS')
                dim_affine = get_transformer_dim('affine')
                dim_cpab = get_transformer_dim('CPAB')

                setup_CPAB_transformer(ncx=self.conf.cpab_tessalation_ncx, ncy=self.conf.cpab_tessalation_ncy, override = True)

                # Prepare transformation layers
                tps_layer = get_transformer_layer('TPS')
                affine_layer = get_transformer_layer('affine')
                cpab_layer = get_transformer_layer('CPAB')

                _1, bias_tps = get_transformer_init_weights(dim_tps, 'TPS')
                _1, bias_affine = get_transformer_init_weights(dim_affine, 'affine')
                _1, bias_cpab = get_transformer_init_weights(dim_cpab, 'CPAB')

                self.theta_tps_t = tf.Variable(initial_value=bias_tps, dtype=tf.float32)
                self.theta_affine_t = tf.Variable(initial_value=bias_affine, dtype=tf.float32)
                self.theta_cpab_t = tf.Variable(initial_value=tf.expand_dims(bias_cpab, 0) + 0.001, dtype=tf.float32)

                # Create grid sampler for guiding image
                # B, H, W, C = (1, self.gi.shape[0], self.gi.shape[1], 3)
                # self.gi_grid = tf.Variable(initial_value=generic_grid_generator(H, W, B))
                # self.gi_grid_inverse = tf.Variable(initial_value=generic_grid_generator(H, W, B))
                # if self.initial_grid is None:
                #     self.initial_grid = tf.Variable(self.gi_grid, trainable=False)

                # Transform matrix for augmenting the guider
                self.augmentation_mat_guider = tf.placeholder(tf.float32, name='augmentation_mat_guider')

                # Transformation output shape
                self.augmentation_output_shape = tf.placeholder(tf.int32, name='augmentation_output_shape')

                # Projective transformation - 8 degrees of freedom
                self.augmentation_mat_guider.set_shape([8])

                # convert guider to feature map using AE
                self.hr_guider_features_t, unet_vars, self.loss_ae = self.reconstruct_using_unet(self.hr_guider_with_shape_t, input_n_channel=self.gi.shape[-1])

                def get_features_guider():
                    return self.hr_guider_features_t

                def get_original_guider():
                    return self.hr_guider_t

                self.hr_guider_features_t = tf.cond(should_deform_and_augment_guider, get_features_guider, get_original_guider)

                def get_deformed_guider():
                    # the shape was lost (changed to unknown), recover it
                    self.hr_guider_features_t.set_shape(self.hr_guider_with_shape_t.get_shape())

                    # TPS / affine transform
                    return tps_layer(
                        cpab_layer(
                            affine_layer(
                                self.hr_guider_features_t, self.theta_affine_t, self.gi.shape[:2]
                            ), self.theta_cpab_t, self.gi.shape[:2]
                        ), self.theta_tps_t, self.gi.shape[:2]
                    )

                self.hr_guider_deformed_t = tf.cond(should_deform_and_augment_guider, get_deformed_guider, get_original_guider)

                def get_augmented_guider():
                    # the shape was lost (changed to unknown), recover it
                    self.hr_guider_deformed_t.set_shape(self.hr_guider_with_shape_t.get_shape())
                    return tf.contrib.image.transform(
                        self.hr_guider_deformed_t, self.augmentation_mat_guider, interpolation='BILINEAR', output_shape=self.augmentation_output_shape
                    )

                self.hr_guider_augmented_t = tf.cond(should_deform_and_augment_guider, get_augmented_guider, get_original_guider)

                self.filters_t_guider = [tf.get_variable(shape=meta.filter_shape_guider[ind], name='filter_guider_%d' % ind,
                                               initializer=tf.random_normal_initializer(
                                                   stddev=np.sqrt(meta.init_variance/np.prod(
                                                       meta.filter_shape_guider[ind][0:3]))))
                               for ind in range(meta.depth_guider)]

                # Define guider layers
                self.layers_t_guider = [self.hr_guider_augmented_t] + [None] * meta.depth_guider

                for l in range(meta.depth_guider - 1):
                    self.layers_t_guider[l + 1] = tf.nn.relu(tf.nn.conv2d(self.layers_t_guider[l], self.filters_t_guider[l],
                                                                [1, 1, 1, 1], "SAME", name='layer_guider_%d' % (l + 1)))
                # Last conv layer (Separate because no ReLU here)
                l = meta.depth_guider - 1

                self.layers_t_guider[l+1] = tf.nn.conv2d(self.layers_t_guider[l], self.filters_t_guider[l],
                                              [1, 1, 1, 1], "SAME", name='layer_guider_%d' % (l + 1))


                # Define the concatenation layer
                #concat_layer = tf.concat([self.lr_son_t, self.layers_t_guider[-1]], 3, name ='concat_layer')
                concat_layer = None


            # Define first layer
            first_layer = concat_layer if concat_layer is not None else self.lr_son_t

            # Filters
            self.filters_t = [tf.get_variable(shape=meta.filter_shape[ind], name='filter_%d' % ind,
                                              initializer=tf.random_normal_initializer(
                                                  stddev=np.sqrt(meta.init_variance/np.prod(
                                                      meta.filter_shape[ind][0:3]))))
                              for ind in range(meta.depth)]

            # Define layers
            self.layers_t = [first_layer] + [None] * meta.depth

            for l in range(meta.depth - 1):
                self.layers_t[l + 1] = tf.nn.relu(tf.nn.conv2d(self.layers_t[l], self.filters_t[l],
                                                               [1, 1, 1, 1], "SAME", name='layer_%d' % (l + 1)))

            # Last conv layer (Separate because no ReLU here)
            l = meta.depth - 1
            self.layers_t[l+1] = tf.nn.conv2d(self.layers_t[l], self.filters_t[l],
                                             [1, 1, 1, 1], "SAME", name='layer_%d' % (l + 1))

            # Output image (Add last conv layer result to input, residual learning with global skip connection)
            self.net_output_before_guider_t = self.layers_t[-1] +  self.conf.learn_residual * self.lr_son_t

            # Loss before guider (L1 loss between label and output layer)
            self.loss_before_guider_t = tf.reduce_mean(tf.reshape(tf.abs(self.net_output_before_guider_t - self.hr_father_t), [-1]))

            # Output image including guider
            self.net_output_t = self.net_output_before_guider_t + self.layers_t_guider[-1]

            # Very final loss
            self.loss_t = tf.reduce_mean(tf.reshape(tf.abs(self.net_output_t - self.hr_father_t), [-1]))

            # Apply adam optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_t)
            guider_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_t * self.conf.learning_rate_guider_ratio)
            tps_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_t * self.conf.learning_rate_tps_ratio)
            affine_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_t * self.conf.learning_rate_affine_ratio)
            cpab_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_t * self.conf.learning_rate_cpab_ratio)

            self.train_op = optimizer.minimize(self.loss_before_guider_t, var_list=self.filters_t)

            # train guider layers and ae layers
            self.train_guider_op = guider_optimizer.minimize(self.loss_t, var_list=unet_vars+self.filters_t_guider)
            self.train_ae_op = guider_optimizer.minimize(self.loss_ae, var_list=unet_vars)

            if self.gi is not None:
                self.train_tps_op = tps_optimizer.minimize(self.loss_t, var_list=[self.theta_tps_t])
                self.train_affine_op = affine_optimizer.minimize(self.loss_t, var_list=[self.theta_affine_t])
                self.train_cpab_op = cpab_optimizer.minimize(self.loss_t, var_list=[self.theta_cpab_t])

            self.init_op = tf.initialize_all_variables()

    def init_sess(self, init_weights=True):
        # Sometimes we only want to initialize some meta-params but keep the weights as they were
        if init_weights:

            # These are for GPU consumption, preventing TF to catch all available GPUs
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            # Initialize computational graph session
            self.sess = tf.Session(graph=self.model, config=config)

            # Initialize weights
            self.sess.run(self.init_op)

        # Initialize all counters etc
        self.loss = [None] * self.conf.max_iters
        self.mse, self.mse_rec, self.psnr_rec, self.interp_mse, self.interp_rec_mse, self.mse_steps = [], [], [], [], [], []
        self.iter = 0
        self.learning_rate = self.conf.learning_rate
        self.learning_rate_change_iter_nums = [0]

    def forward_backward_pass(self, lr_son, hr_father, hr_guider, augmentation_mat_guider):
        # First gate for the lr-son into the network is interpolation to the size of the father
        # Note: we specify both output_size and scale_factor. best explained by example: say father size is 9 and sf=2,
        # small_son size is 4. if we upscale by sf=2 we get wrong size, if we upscale to size 9 we get wrong sf.
        # The current imresize implementation supports specifying both.
        interpolated_lr_son = np.clip(imresize(lr_son, self.sf, hr_father.shape, self.conf.upscale_method), 0, 1)

        # [GUY] add n_channels when needed
        interpolated_lr_son, hr_father =  add_n_channels_dim(interpolated_lr_son, hr_father)
        if hr_guider is not None:
            hr_guider, = add_n_channels_dim(hr_guider)


        if hr_guider is not None:
            feed_dict = {
                'learning_rate:0': self.learning_rate,
                'lr_son:0': np.expand_dims(interpolated_lr_son, 0),
                'hr_father:0': np.expand_dims(hr_father, 0),
                'hr_guider:0': np.expand_dims(hr_guider, 0),
                'hr_guider_with_shape:0': np.expand_dims(self.gi, 0),
                'augmentation_mat_guider:0': augmentation_mat_guider,
                'augmentation_output_shape:0': interpolated_lr_son.shape[:2]
            }
            fetch_args = [self.train_op, self.train_guider_op, self.train_tps_op, self.train_affine_op, self.train_cpab_op, self.hr_guider_augmented_t, self.hr_guider_deformed_t, self.loss_t, self.net_output_t]

            # train unet to reconstruct only for few iterations
            fetch_args = [self.train_ae_op] * (self.iter < self.conf.train_ae_iters) + fetch_args
            *_, self.hr_guider_augmented, self.hr_guider_deformed, self.loss[self.iter], train_output = \
                self.sess.run(
                    fetch_args, feed_dict
                )

        else:
            feed_dict = {
                'learning_rate:0': self.learning_rate,
                'lr_son:0': np.expand_dims(interpolated_lr_son, 0),
                'hr_father:0': np.expand_dims(hr_father, 0),
            }
            _1, self.loss[self.iter], train_output = \
                self.sess.run(
                    [self.train_op, self.loss_t, self.net_output_t], feed_dict
                )

        return np.clip(np.squeeze(train_output), 0, 1)

    def forward_pass(self, lr_son, hr_guider, hr_father_shape=None, augmentation_mat_guider=None, should_deform_and_augment_guider=True):
        # First gate for the lr-son into the network is interpolation to the size of the father
        interpolated_lr_son = imresize(lr_son, self.sf, hr_father_shape, self.conf.upscale_method)

        # [GUY] add n_channels when needed
        interpolated_lr_son, = add_n_channels_dim(interpolated_lr_son)
        if hr_guider is not None:
            hr_guider, = add_n_channels_dim(hr_guider)

            # could be 1 ratio (like when running on full size input with full size guider)
            guider_to_im_ratio = np.true_divide(hr_guider.shape[:2], hr_father_shape[:2])

            # if no augmentation sent, only perform downscaling
            augmentation_mat_guider = augmentation_mat_guider or np.array([guider_to_im_ratio[0], 0, 0, 0, guider_to_im_ratio[1], 0, 0, 0])

            # Create feed dict
            # in most cases hr_guider=self.gi, in final_test hr_guider was
            # rotated so hr_guider!=self.gi
            feed_dict = {'lr_son:0': np.expand_dims(interpolated_lr_son, 0),
                         'hr_guider:0': np.expand_dims(hr_guider, 0) if hr_guider is not None else None,
                         'hr_guider_with_shape:0': np.expand_dims(self.gi, 0) if hr_guider is not None else None,
                         'augmentation_mat_guider:0': augmentation_mat_guider,
                         'augmentation_output_shape:0': interpolated_lr_son.shape[:2],
                         'should_deform_and_augment_guider:0': should_deform_and_augment_guider,
                         }
        else:
            feed_dict = {
                'lr_son:0': np.expand_dims(interpolated_lr_son, 0),
            }

        # Run network
        return np.clip(np.squeeze(self.sess.run([self.net_output_t], feed_dict)), 0, 1)

    def learning_rate_policy(self):
        # fit linear curve and check slope to determine whether to do nothing, reduce learning rate or finish
        if (not (1 + self.iter) % self.conf.learning_rate_policy_check_every
                and self.iter - self.learning_rate_change_iter_nums[-1] > self.conf.min_iters):
            # noinspection PyTupleAssignmentBalance
            [slope, _], [[var, _], _] = np.polyfit(self.mse_steps[-int(self.conf.learning_rate_slope_range /
                                                                    self.conf.run_test_every):],
                                                   self.mse_rec[-int(self.conf.learning_rate_slope_range /
                                                                  self.conf.run_test_every):],
                                                   1, cov=True)

            # We take the the standard deviation as a measure
            std = np.sqrt(var)

            # Verbose
            print('slope: ', slope, 'STD: ', std)

            # Determine learning rate maintaining or reduction by the ration between slope and noise
            if -self.conf.learning_rate_change_ratio * slope < std:
                self.learning_rate /= 10
                print("learning rate updated: ", self.learning_rate)

                # Keep track of learning rate changes for plotting purposes
                self.learning_rate_change_iter_nums.append(self.iter)

    def quick_test(self):
        # There are four evaluations needed to be calculated:

        # Run net on the input to get the output super-resolution (almost final result, only post-processing needed)
        self.sr = self.forward_pass(self.input, self.gi, self.output_shape if self.gi is not None else None)
        # self.sr = self.forward_pass(self.input, self.gi, self.gi_per_sf.shape if self.gi is not None else None)

        # 1. True MSE (only if ground-truth was given), note: this error is before post-processing.

        #TODO: do something less ugly
        if self.gt_per_sf is not None and self.gt_per_sf.shape[2] == 1:
            self.gt_per_sf, = remove_n_channels_dim(self.gt_per_sf)
        self.mse = self.mse + [np.mean(np.ndarray.flatten(np.square(self.gt_per_sf - self.sr)))
                    if self.gt_per_sf is not None else None]

        # 2. Reconstruction MSE, run for reconstruction- try to reconstruct the input from a downscaled version of it
        self.reconstruct_output = self.forward_pass(self.father_to_son(self.input), self.gi, self.input.shape)
        # self.reconstruct_output = self.forward_pass(self.father_to_son(self.input), self.father_to_son(self.gi), self.input.shape)

        # [GUY] add n_channels when needed
        self.input, self.reconstruct_output, self.sr, self.gt_per_sf = add_n_channels_dim(self.input, self.reconstruct_output, self.sr, self.gt_per_sf)

        self.mse_rec.append(np.mean(np.ndarray.flatten(np.square(self.input - self.reconstruct_output))))

        # 2.5 [GUY] Reconstruction PSNR
        with tf.Session():
            self.psnr_rec.append(tf.image.psnr(self.input, self.reconstruct_output, max_val=1).eval())
            self.psnr.append(tf.image.psnr(self.gt_per_sf, self.sr, max_val=1).eval() if self.gt_per_sf is not None else None)

        # 3. True MSE of simple interpolation for reference (only if ground-truth was given)
        interp_sr = np.clip(imresize(self.input, self.sf, self.output_shape, self.conf.upscale_method),0 ,1)
        self.interp_mse = (self.interp_mse + [np.mean(np.ndarray.flatten(np.square(self.gt_per_sf - interp_sr)))
                           if self.gt_per_sf is not None else None])

        # 4. Reconstruction MSE of simple interpolation over downscaled input
        interp_rec = np.clip(imresize(self.father_to_son(self.input), self.sf, self.input.shape[0:2], self.conf.upscale_method),0 ,1)
        self.interp_rec_mse.append(np.mean(np.ndarray.flatten(np.square(self.input - interp_rec))))

        # Track the iters in which tests are made for the graphics x axis
        self.mse_steps.append(self.iter)

        # Display test results if indicated
        if self.conf.display_test_results:
            print('iteration: ', self.iter,
                  'reconstruct mse:', self.mse_rec[-1],
                  'true mse:', (self.mse[-1]),
                  'reconstruct psnr:', self.psnr_rec[-1],
                  'true_psnr:', self.psnr[-1]
                  if self.mse else None)

        # plot losses if needed
        if self.conf.plot_losses:
            self.plot()

    def train(self):
        # main training loop
        for self.iter in range(self.conf.max_iters):
            # Use augmentation from original input image to create current father.
            # If other scale factors were applied before, their result is also used (hr_fathers_in)

            chosen_image, chosen_augmentation, chosen_augmentation_guider = random_augment(
                ims=self.hr_fathers_sources,
                guiding_im_shape=self.gi.shape if self.gi is not None else None,
                base_scales=[1.0] + self.conf.scale_factors,
                leave_as_is_probability=self.conf.augment_leave_as_is_probability,
                no_interpolate_probability=self.conf.augment_no_interpolate_probability,
                min_scale=self.conf.augment_min_scale,
                max_scale=([1.0] + self.conf.scale_factors)[len(self.hr_fathers_sources)-1],
                allow_rotation=self.conf.augment_allow_rotation,
                scale_diff_sigma=self.conf.augment_scale_diff_sigma,
                shear_sigma=self.conf.augment_shear_sigma,
                crop_size=self.conf.crop_size)

            self.hr_father = tf.contrib.image.transform(
                chosen_image, chosen_augmentation, interpolation='BILINEAR', output_shape=(self.conf.crop_size, self.conf.crop_size)
            ).eval(session=tf.Session())

            # Get lr-son from hr-father
            self.lr_son = self.father_to_son(self.hr_father)

            # our guider is always the guiding image (except for final_test)
            self.hr_guider = self.gi

            # run network forward and back propagation, one iteration (This is the heart of the training)
            self.train_output = self.forward_backward_pass(self.lr_son, self.hr_father, self.hr_guider, chosen_augmentation_guider)
            # Display info and save weights
            if not self.iter % self.conf.display_every:
                print(
                    'sf:', self.sf*self.base_sf, ', iteration: ', self.iter,
                    ', loss: ', self.loss[self.iter],
                )

            # Test network
            if self.conf.run_test and (not self.iter % self.conf.run_test_every):
                self.quick_test()

            # Consider changing learning rate or stop according to iteration number and losses slope
            self.learning_rate_policy()

            # stop when minimum learning rate was passed
            if self.learning_rate < self.conf.min_learning_rate:
                break

    def father_to_son(self, hr_father):
        if hr_father is None:
            return None

        # Create son out of the father by downscaling and if indicated adding noise
        lr_son = imresize(hr_father, 1.0 / self.sf, kernel=self.kernel)
        return np.clip(lr_son + np.random.randn(*lr_son.shape) * self.conf.noise_std, 0, 1)

    def final_test(self):
        # Run over 8 augmentations of input - 4 rotations and mirror (geometric self ensemble)
        # Rotate 90*k degrees and mirror flip when k>=4

        outputs = []

        def rotate_and_flip(input, k, grid=None):
            # rotate and flip the input
            rotated_input =  np.rot90(input, k) \
                if k < 4 else np.fliplr(np.rot90(input, k))

            # rotate the grid if needed
            if grid is not None:
                # swap axes to match for format of tf.rot90
                swapped_grid = tf.transpose(grid,[2,3,1,0])[:,:,:,0]
                # rotate and flip
                rotated_grid = tf.image.rot90(swapped_grid, k)  \
                    if k < 4 else tf.image.flip_left_right(tf.image.rot90(swapped_grid, k))
                # swap axes back
                rotated_grid = tf.expand_dims(tf.transpose(rotated_grid,[2,0,1]),0)

            return rotated_input, rotated_grid if grid is not None else None

        # deform the guiding image
        deformed_gi = self.sess.run(
            [self.hr_guider_deformed_t], {
                'hr_guider:0': np.expand_dims(self.gi, 0),
                'hr_guider_with_shape:0': np.expand_dims(self.gi, 0),
                'augmentation_mat_guider:0': np.array([1, 0, 0, 0, 1, 0, 0, 0]),
                'augmentation_output_shape:0': self.gi.shape[:2]

            }
        )[0] if self.gi is not None else None

        # The weird range means we only do it once if output_flip is disabled
        # We need to check if scale factor is symmetric to all dimensions, if not we will do 180 jumps rather than 90
        for k in range(0, 1 + 7 * self.conf.output_flip, 1 + int(self.sf[0] != self.sf[1])):

            if self.gi is not None:
                B, H, W, C = (1, self.gi.shape[0], self.gi.shape[1], 3)
                sampler_grid = generic_grid_generator(H, W, B)
                test_input, test_grid = rotate_and_flip(self.input, k, grid=sampler_grid if self.gi is not None else None)
                augmented_gi = generic_transformer(
                        deformed_gi, test_grid
                ).eval(session=tf.Session())[0]

                # scale to deformed_gi to the right sf
                # TODO: should we clip?
                augmented_gi = imresize(augmented_gi,
                                        scale_factor=self.sf*self.base_sf/self.conf.scale_factors[-1],
                                        kernel=self.conf.downscale_gt_method)


            # Apply network on the rotated input
            tmp_output = self.forward_pass(
                test_input, augmented_gi if self.gi is not None else None,
                hr_father_shape=augmented_gi.shape if self.gi is not None else None,
                should_deform_and_augment_guider=False # as we just augmented the guider here
            )

            # Undo the rotation for the processed output (mind the opposite order of the flip and the rotation)
            tmp_output = np.rot90(tmp_output, -k) if k < 4 else np.rot90(np.fliplr(tmp_output), -k)

            # fix SR output with back projection technique for each augmentation

            for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
                tmp_output = back_projection(add_n_channels_dim(tmp_output)[0], self.input, down_kernel=self.kernel,
                                             up_kernel=self.conf.upscale_method, sf=self.sf)

            # save outputs from all augmentations
            outputs.append(tmp_output)

        # Take the median over all 8 outputs
        almost_final_sr = np.median(outputs, 0)

        # Again back projection for the final fused result
        for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
            almost_final_sr = back_projection(add_n_channels_dim(almost_final_sr)[0], self.input, down_kernel=self.kernel,
                                              up_kernel=self.conf.upscale_method, sf=self.sf)

        # Now we can keep the final result (in grayscale case, colors still need to be added, but we don't care
        # because it is done before saving and for every other purpose we use this result)
        self.final_sr = almost_final_sr

        # Add colors to result image in case net was activated only on grayscale
        return self.final_sr

    def base_change(self):
        # If there is no base scale large than the current one get out of here
        if len(self.conf.base_change_sfs) < self.base_ind + 1:
            return

        # Change base input image if required (this means current output becomes the new input)
        if sum(abs(np.array(self.conf.scale_factors[self.sf_ind]) - np.array(self.conf.base_change_sfs[self.base_ind]))) < 0.001:
            if len(self.conf.base_change_sfs) > self.base_ind:

                # The new input is the current output
                self.input = self.final_sr

                # The new base scale_factor
                self.base_sf = self.conf.base_change_sfs[self.base_ind]

                # Keeping track- this is the index inside the base scales list (provided in the config)
                self.base_ind += 1

            base_sf_str = ''.join('X%.2f' % s for s in self.base_sf)
            print('base changed to %s' % base_sf_str)

    def plot(self):
        plots_data, labels = zip(*[(np.array(x), l) for (x, l)
                                   in zip([self.mse, self.mse_rec, self.interp_mse, self.interp_rec_mse],
                                          ['True MSE', 'Reconstruct MSE', 'Bicubic to ground truth MSE',
                                           'Bicubic to reconstruct MSE']) if x is not None])

        # For the first iteration create the figure
        if not self.iter and not self.sf_ind:
            # Create figure and split it using GridSpec. Name each region as needed
            self.fig = plt.figure(figsize=(9.5, 9))
            grid = GridSpec(4, 4)
            self.loss_plot_space = plt.subplot(grid[:-1, :])
            self.lr_son_image_space = plt.subplot(grid[3, 0])
            self.hr_father_image_space = plt.subplot(grid[3, 3])
            self.out_image_space = plt.subplot(grid[3, 1])
            if self.hr_guider is not None:
                self.hr_guider_image_space = plt.subplot(grid[3, 2])

            # Activate interactive mode for live plot updating
            plt.ion()

            # Set some parameters for the plots
            self.loss_plot_space.set_xlabel('step')
            self.loss_plot_space.set_ylabel('MSE')
            self.loss_plot_space.grid(True)
            self.loss_plot_space.set_yscale('log')
            self.loss_plot_space.legend()
            self.plots = [None] * 4

            # loop over all needed plot types. if some data is none than skip, if some data is one value tile it
            self.plots = self.loss_plot_space.plot(*[[0]] * 2 * len(plots_data))

        # Update plots
        for plot, plot_data in zip(self.plots, plots_data):
            plot.set_data(self.mse_steps, plot_data)

            self.loss_plot_space.set_xlim([0, self.iter + 1])
            all_losses = np.array(plots_data)

            # convert None values to np.nan
            all_losses = np.array([[loss or np.nan for loss in losses] for losses in all_losses])

            self.loss_plot_space.set_ylim([np.nanmin(all_losses)*0.9, np.nanmax(all_losses)*1.1])

        # Mark learning rate changes
        for iter_num in self.learning_rate_change_iter_nums:
            self.loss_plot_space.axvline(iter_num)

        # Add legend to graphics
        self.loss_plot_space.legend(labels)

        # Show current input and output images
        self.lr_son, self.hr_father = remove_n_channels_dim(self.lr_son, self.hr_father)
        self.lr_son_image_space.imshow(self.lr_son, vmin=0.0, vmax=1.0, cmap=self.conf.cmap)
        self.out_image_space.imshow(self.train_output, vmin=0.0, vmax=1.0, cmap=self.conf.cmap)
        self.hr_father_image_space.imshow(self.hr_father, vmin=0.0, vmax=1.0, cmap=self.conf.cmap)

        if self.hr_guider is not None:
            self.hr_guider_image_space.imshow(self.hr_guider_augmented[0], vmin=0.0, vmax=1.0, cmap=self.conf.cmap)

        # These line are needed in order to see the graphics at real time
        self.fig.canvas.draw()
        plt.pause(1)

    def create_displacement_map(self):
        # create identity grid
        B, H, W = (1, self.gi.shape[0], self.gi.shape[1])
        identity_grid = generic_grid_generator(H, W, B)
        identity_grid = np.transpose(identity_grid, [0,2,3,1])
        zeros = np.zeros((1, H, W, 1))
        identity_grid = np.concatenate([identity_grid, zeros], 3)

        # scale identity_grid to [0,1]
        identity_grid += 1
        identity_grid /= 2

        # deform it with the same transformation
        deformed_grid = self.sess.run(
            [self.hr_guider_deformed_t],
            {
                'hr_guider:0': identity_grid,
                'hr_guider_with_shape:0': identity_grid,
                'augmentation_mat_guider:0': np.array([1, 0, 0, 0, 1, 0, 0, 0]),
                'augmentation_output_shape:0': self.gi.shape[:2]
            }
        )[0]

        # get the diff
        # diff_grid = np.abs(deformed_grid - identity_grid)[:,:,:,:2]
        diff_grid = (deformed_grid - identity_grid)[:,:,:,:2]

        # concatenate zeros for the third axis, we only use the first two (X/Y)
        displacement_map = np.concatenate([diff_grid, zeros],3)[0]

        # scale the values
        displacement_map *= 256

        # displacement_map[:,:,0] -= np.min(displacement_map[:,:,0])
        # displacement_map[:,:,0] /= (np.max(displacement_map[:,:,0]) / 255)

        # scale the Y values to 0-255
        # displacement_map[:,:,1] -= np.min(displacement_map[:,:,1])
        # displacement_map[:,:,1] /= (np.max(displacement_map[:,:,1]) / 255)

        return displacement_map

    def reconstruct_using_unet(self, input, input_n_channel):
        output, variables, _ = create_conv_net(input, 0.8, 3, 3)
        return output, variables, tf.nn.l2_loss(output-input)
