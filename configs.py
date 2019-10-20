import os
import matplotlib.pyplot as plt


class Config:
    # network meta params
    python_path = '/home/assafsho/PycharmProjects/network/venv/bin/python2.7'
    scale_factors = [[2.0, 2.0]]  # list of pairs (vertical, horizontal) for gradual increments in resolution
    base_change_sfs = []  # list of scales after which the input is changed to be the output (recommended for high sfs)
    max_iters = 3000
    min_iters = 256
    train_ae_iters = 128
    min_learning_rate = 9e-6  # this tells the algorithm when to stop (specify lower than the last learning-rate)
    output_flip = True  # geometric self-ensemble (see paper)
    downscale_method = 'cubic'  # a string ('cubic', 'linear'...), has no meaning if kernel given
    upscale_method = 'cubic'  # this is the base interpolation from which we learn the residual (same options as above)
    downscale_gt_method = 'cubic'  # when ground-truth given and intermediate scales tested, we shrink gt to wanted size
    learn_residual = True  # when true, we only learn the residual from base interpolation
    init_variance = 0.1  # variance of weight initializations, typically smaller when residual learning is on
    back_projection_iters = [10]  # for each scale num of bp iterations (same length as scale_factors)
    random_crop = True
    crop_size = 128
    noise_std = 0.0  # adding noise to lr-sons. small for real images, bigger for noisy images and zero for ideal case
    init_net_for_each_sf = False  # for gradual sr- should we optimize from the last sf or initialize each time?

    # Params concerning learning rate policy
    learning_rate = 0.0001
    # learning_rate_grid = 0.0001
    # learning_rate_grid = 0.001
    learning_rate_change_ratio = 1.5  # ratio between STD and slope of linear fit, under which lr is reduced
    learning_rate_policy_check_every = 60
    learning_rate_slope_range = 256

    # Data augmentation related params
    augment_leave_as_is_probability = 0.05
    augment_no_interpolate_probability = 0.45
    augment_min_scale = 0.5
    augment_scale_diff_sigma = 0.25
    augment_shear_sigma = 0.1
    augment_allow_rotation = True  # recommended false for non-symmetric kernels

    # params related to test and display
    run_test = True
    run_test_every = 50
    display_every = 5
    name = 'test'
    plot_losses = False
    result_path = os.path.dirname(__file__) + '/results'
    create_results_dir = True
    input_path = local_dir = os.path.dirname(__file__) + '/test_data'
    create_code_copy = True  # save a copy of the code in the results folder to easily match code changes to results
    display_test_results = True
    save_results = True
    cmap = None
    img_ext = 'png'
    guiding_img_ext = 'png'

    # params related to deformation
    learning_rate_cpab_ratio = 1.0 # ratio between lr and deformation lr
    learning_rate_affine_ratio = 1.0 # ratio between lr and deformation lr
    learning_rate_tps_ratio = 1.0 # ratio between lr and deformation lr
    learning_rate_guider_ratio = 1.0 # ratio between lr and guider lr
    cpab_tessalation_ncx = 2
    cpab_tessalation_ncy = 2

    def __init__(self, input_filter_depth=3, output_filter_depth=3,
                 guider_filter_depth=3, guider_output_filter_depth=3,
                 width=64, depth=8, depth_guider=8):
        self.width = width
        self.depth = depth
        self.depth_guider = depth_guider
        # network meta params that by default are determined (by other params) by other params but can be changed
        self.filter_shape = ([[3, 3, input_filter_depth, self.width]] +
                             [[3, 3, self.width, self.width]] * (self.depth-2) +
                             [[3, 3, self.width, output_filter_depth]])
        self.filter_shape_guider = ([[1, 1, guider_filter_depth, self.width]] +
                             [[1, 1, self.width, self.width]] * (self.depth_guider-2) +
                             [[1, 1, self.width, guider_output_filter_depth]])


########################################
# Some pre-made useful example configs #
########################################

# Basic default config (same as not specifying), non-gradual SRx2 with default bicubic kernel (Ideal case)
# example is set to run on set14
X2_ONE_JUMP_IDEAL_CONF = Config()
X2_ONE_JUMP_IDEAL_CONF.input_path = os.path.dirname(__file__) + '/set14'

# [GUY]
THERMAL_IMAGES_CONF = Config(input_filter_depth=3, output_filter_depth=3, guider_output_filter_depth=3,
                             depth_guider=3)
THERMAL_IMAGES_CONF.plot_losses = True
THERMAL_IMAGES_CONF.crop_size = 48
THERMAL_IMAGES_CONF.max_iters = 800
THERMAL_IMAGES_CONF.train_ae_iters = 128
THERMAL_IMAGES_CONF.run_test_every = 20
THERMAL_IMAGES_CONF.display_every = 1
# THERMAL_IMAGES_CONF.input_path = os.path.dirname(__file__) + '/data_processed/current3'
THERMAL_IMAGES_CONF.input_path = os.path.dirname(__file__) + '/ULB17-VT'
# THERMAL_IMAGES_CONF.img_ext = 'tiff'
# THERMAL_IMAGES_CONF.guiding_img_ext = 'jpg'
#THERMAL_IMAGES_CONF.scale_factors = [[2.0, 2.0], [4.0, 4.0]]
# THERMAL_IMAGES_CONF.input_path = os.path.dirname(__file__) + '/Maagad_reg2'
THERMAL_IMAGES_CONF.img_ext = 'png'
THERMAL_IMAGES_CONF.guiding_img_ext = 'png'
THERMAL_IMAGES_CONF.scale_factors = [[2.0, 2.0], [4.0, 4.0]]  # list of pairs (vertical, horizontal) for gradual increments in resolution
THERMAL_IMAGES_CONF.back_projection_iters = [6, 10]
THERMAL_IMAGES_CONF.base_change_sfs = [[2.0, 2.0]]  # list of pairs (vertical, horizontal) for gradual increments in resolution
#THERMAL_IMAGES_CONF.scale_factors = [[4.0, 4.0]]  # list of pairs (vertical, horizontal) for gradual increments in resolution
THERMAL_IMAGES_CONF.learning_rate_cpab_ratio = 1
THERMAL_IMAGES_CONF.learning_rate_affine_ratio = 2
THERMAL_IMAGES_CONF.learning_rate_tps_ratio = 0.1
THERMAL_IMAGES_CONF.learning_rate_guider_ratio = 1
THERMAL_IMAGES_CONF.cpab_tessalation_ncx = 4
THERMAL_IMAGES_CONF.cpab_tessalation_ncy = 4

# THERMAL_IMAGES_CONF.scale_factors = [[1.0, 1.5], [1.5, 1.0], [1.5, 1.5], [1.5, 2.0], [2.0, 1.5], [2.0, 2.0]]
# THERMAL_IMAGES_CONF.back_projection_iters = [6, 6, 8, 10, 10, 12]
# THERMAL_IMAGES_CONF.noise_std = 0.05  # adding noise to lr-sons. small for real images, bigger for noisy images and zero for ideal case

# [GUY]
DEPTH_MAPS_CONF = Config(input_filter_depth=1,
                         output_filter_depth=1,
                         guider_output_filter_depth=1)
DEPTH_MAPS_CONF.input_path = os.path.dirname(__file__) + '/vase'
DEPTH_MAPS_CONF.img_ext = 'png'
DEPTH_MAPS_CONF.guiding_img_ext = 'png'
DEPTH_MAPS_CONF.cmap = 'gray'
DEPTH_MAPS_CONF.max_iters = 800
DEPTH_MAPS_CONF.scale_factors = [[2.0, 2.0], [4.0, 4.0]]  # list of pairs (vertical, horizontal) for gradual increments in resolution
DEPTH_MAPS_CONF.back_projection_iters = [6, 6]
# DEPTH_MAPS_CONF.base_change_sfs = [[2.0, 2.0]]  # list of pairs (vertical, horizontal) for gradual increments in resolution
# DEPTH_MAPS_CONF.scale_factors = [[2.0, 2.0], [4.0, 4.5]]  # list of pairs (vertical, horizontal) for gradual increments in resolution
# DEPTH_MAPS_CONF.scale_factors = [[4.0, 4.0]]
# DEPTH_MAPS_CONF.scale_factors = [[1.0, 2.0], [2.0, 1.0], [2.0, 2.0], [2.0, 3.0], [3.0, 2.0], [3.0, 3.0], [3.0, 4.0], [4.0, 3.0], [4.0, 4.0], [4.0, 4.5]]
# DEPTH_MAPS_CONF.back_projection_iters = [6, 6, 6, 8, 8, 8, 10, 10, 12, 12]
DEPTH_MAPS_CONF.run_test_every = 20
DEPTH_MAPS_CONF.display_every = 1
DEPTH_MAPS_CONF.plot_losses = True
DEPTH_MAPS_CONF.crop_size = 128
# DEPTH_MAPS_CONF.init_net_for_each_sf = True

# Same as above but with visualization (Recommended for one image, interactive mode, for debugging)
X2_IDEAL_WITH_PLOT_CONF = Config()
X2_IDEAL_WITH_PLOT_CONF.plot_losses = True
X2_IDEAL_WITH_PLOT_CONF.run_test_every = 20
X2_IDEAL_WITH_PLOT_CONF.input_path = os.path.dirname(__file__) + '/example_with_gt'

# Gradual SRx2, to achieve superior results in the ideal case
X2_GRADUAL_IDEAL_CONF = Config()
X2_GRADUAL_IDEAL_CONF.scale_factors = [[1.0, 1.5], [1.5, 1.0], [1.5, 1.5], [1.5, 2.0], [2.0, 1.5], [2.0, 2.0]]
X2_GRADUAL_IDEAL_CONF.back_projection_iters = [6, 6, 8, 10, 10, 12]
X2_GRADUAL_IDEAL_CONF.input_path = os.path.dirname(__file__) + '/set14'

# Applying a given kernel. Rotations are canceled sense kernel may be non-symmetric
X2_GIVEN_KERNEL_CONF = Config()
X2_GIVEN_KERNEL_CONF.output_flip = False
X2_GIVEN_KERNEL_CONF.augment_allow_rotation = False
X2_GIVEN_KERNEL_CONF.back_projection_iters = [2]
X2_GIVEN_KERNEL_CONF.input_path = os.path.dirname(__file__) + '/kernel_example'

# An example for a typical setup for real images. (Kernel needed + mild unknown noise)
# back-projection is not recommended because of the noise.
X2_REAL_CONF = Config()
X2_REAL_CONF.output_flip = False
X2_REAL_CONF.back_projection_iters = [0]
X2_REAL_CONF.input_path = os.path.dirname(__file__) + '/real_example'
X2_REAL_CONF.noise_std = 0.0125
X2_REAL_CONF.augment_allow_rotation = False
X2_REAL_CONF.augment_scale_diff_sigma = 0
X2_REAL_CONF.augment_shear_sigma = 0
X2_REAL_CONF.augment_min_scale = 0.75
