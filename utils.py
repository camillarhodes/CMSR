import numpy as np
from math import pi, sin, cos
# from cv2 import Canny
from imresize import imresize
from shutil import copy
from time import strftime, localtime
import os
import glob
from scipy.ndimage import measurements, interpolation
from scipy.io import loadmat


def random_augment(ims,
                   guiding_im_shape=None,
                   base_scales=None,
                   leave_as_is_probability=0.2,
                   no_interpolate_probability=0.3,
                   min_scale=0.5,
                   max_scale=1.0,
                   allow_rotation=True,
                   scale_diff_sigma=0.01,
                   shear_sigma=0.01,
                   crop_size=128,
                   should_train_guider=False):
    """Takes a random crop of the image and the guiding image.
    Returns:
        1. the image chosen randomly from `ims` list
        3. the chosen augmentation
        4. the augmentation (from 3) with additional downscaling to be used on the guider/grid
    """

    # Determine which kind of augmentation takes place according to probabilities
    random_chooser = np.random.rand()

    # Option 1: No augmentation, return the original image
    if random_chooser < leave_as_is_probability:
        mode = 'leave_as_is'

    # Option 2: Only non-interpolated augmentation, which means 8 possible augs (4 rotations X 2 mirror flips)
    elif leave_as_is_probability < random_chooser < leave_as_is_probability + no_interpolate_probability:
        mode = 'no_interp'

    # Option 3: Affine transformation (uses interpolation)
    else:
        mode = 'affine'

    # If scales not given, calculate them according to sizes of images. This would be suboptimal, because when scales
    # are not integers, different scales can have the same image shape.
    if base_scales is None:
        base_scales = [np.sqrt(np.prod(im.shape) / np.prod(ims[0].shape)) for im in ims]

    # In case scale is a list of scales with take the smallest one to be the allowed minimum
    max_scale = np.min([max_scale])

    if should_train_guider:
        min_scale += 3
        max_scale += 3

    # Determine a random scale by probability
    if mode == 'leave_as_is':
        scale = 1.0
    else:
        scale = np.random.rand() * (max_scale - min_scale) + min_scale

    # The image we will use is the smallest one that is bigger than the wanted scale
    # (Using a small value overlap instead of >= to prevent float issues)
    scale_ind, base_scale = next((ind, np.min([base_scale])) for ind, base_scale in enumerate(base_scales)
                                 if np.min([base_scale]) > scale - 1.0e-6)
    im = ims[scale_ind]

    # Next are matrices whose multiplication will be the transformation. All are 3x3 matrices.

    # First matrix shifts image to center so that crop is in the center of the image
    shift_to_center_mat = np.array([[1, 0, - im.shape[1] / 2.0],
                                    [0, 1, - im.shape[0] / 2.0],
                                    [0, 0, 1]])

    shift_back_from_center = np.array([[1, 0, im.shape[1] / 2.0],
                                       [0, 1, im.shape[0] / 2.0],
                                       [0, 0, 1]])

    # Keeping the transform interpolation free means only shifting by integers
    if mode != 'affine':
        shift_to_center_mat = np.round(shift_to_center_mat)
        shift_back_from_center = np.round(shift_back_from_center)

    # Scale matrix. if affine, first determine global scale by probability, then determine difference between x scale
    # and y scale by gaussian probability.
    if mode == 'affine':
        scale /= base_scale
        scale_diff = np.random.randn() * scale_diff_sigma
    else:
        scale = 1.0
        scale_diff = 0.0
    # In this matrix we also incorporate the possibility of mirror reflection (unless leave_as_is).
    if mode == 'leave_as_is' or not allow_rotation:
        reflect = 1
    else:
        reflect = np.sign(np.random.randn())

    scale_mat = np.array([[reflect * (scale + scale_diff / 2), 0, 0],
                          [0, scale - scale_diff / 2, 0],
                          [0, 0, 1]])

    # Shift matrix, this actually creates the random crop
    shift_x = np.random.rand() * np.clip(scale * im.shape[1] - crop_size, 0, 9999)
    shift_y = np.random.rand() * np.clip(scale * im.shape[0] - crop_size, 0, 9999)
    shift_mat = np.array([[1, 0, - shift_x],
                          [0, 1, - shift_y],
                          [0, 0, 1]])

    # Keeping the transform interpolation free means only shifting by integers
    if mode != 'affine':
        shift_mat = np.round(shift_mat)

    # Rotation matrix angle. if affine, set a random angle. if no_interp then theta can only be pi/2 times int.
    if mode == 'affine':
        theta = np.random.rand() * 2 * pi
    elif mode == 'no_interp':
        theta = np.random.randint(4) * pi / 2
    else:
        theta = 0
    if not allow_rotation:
        theta = 0

    # Rotation matrix structure
    rotation_mat = np.array([[cos(theta), sin(theta), 0],
                             [-sin(theta), cos(theta), 0],
                             [0, 0, 1]])

    # Shear Matrix, only for affine transformation.
    if mode == 'affine':
        shear_x = np.random.randn() * shear_sigma
        shear_y = np.random.randn() * shear_sigma
    else:
        shear_x = shear_y = 0
    shear_mat = np.array([[1, shear_x, 0],
                          [shear_y, 1, 0],
                          [0, 0, 1]])

    # Create the final transformation by multiplying all the transformations.
    augmentation_mat = (shift_back_from_center
                     .dot(shift_mat)
                     .dot(shear_mat)
                     .dot(rotation_mat)
                     .dot(scale_mat)
                     .dot(shift_to_center_mat))

    if guiding_im_shape:
        guider_to_im_ratio = np.true_divide(guiding_im_shape, im.shape)[:2]

        # first scale the guider/grid to the size of the image
        scale_guider_mat = np.array([[1.0 / guider_to_im_ratio[0], 0, 0],
                                     [0, 1.0 / guider_to_im_ratio[1], 0],
                                     [0, 0, 1]])

        # then perform the same augmentation
        augmentation_mat_guider = augmentation_mat.dot(scale_guider_mat)


        return im, flatten_transform(augmentation_mat), flatten_transform(augmentation_mat_guider)

    return im, flatten_transform(augmentation_mat), None


def flatten_transform(transformation_mat):
        """This converts the augmentation matrix into the format of
        tf.contrib.image.transform"""
        return np.linalg.inv(transformation_mat).flatten()[:8]


def back_projection(y_sr, y_lr, down_kernel, up_kernel, sf=None):
    y_sr += imresize(y_lr - imresize(y_sr,
                                     scale_factor=1.0/sf,
                                     output_shape=y_lr.shape,
                                     kernel=down_kernel),
                     scale_factor=sf,
                     output_shape=y_sr.shape,
                     kernel=up_kernel)
    return np.clip(y_sr, 0, 1)


def preprocess_kernels(kernels, conf):
    # Load kernels if given files. if not just use the downscaling method from the configs.
    # output is a list of kernel-arrays or a a list of strings indicating downscaling method.
    # In case of arrays, we shift the kernels (see next function for explanation why).
    # Kernel is a .mat file (MATLAB) containing a variable called 'Kernel' which is a 2-dim matrix.
    if kernels is not None:
        return [kernel_shift(loadmat(kernel)['Kernel'], sf)
                for kernel, sf in zip(kernels, conf.scale_factors)]
    else:
        return [conf.downscale_method] * len(conf.scale_factors)


def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel:
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)


def prepare_result_dir(conf):
    # Create results directory
    if conf.create_results_dir:
        conf.result_path += '/' + conf.name + strftime('_%b_%d_%H_%M_%S', localtime())
        os.makedirs(conf.result_path)

    # Put a copy of all *.py files in results path, to be able to reproduce experimental results
    if conf.create_code_copy:
        local_dir = os.path.dirname(__file__)
        for py_file in glob.glob(local_dir + '/*.py'):
            copy(py_file, conf.result_path)

    return conf.result_path


def add_n_channels_dim(*images):
    # Adds a n_channels dim if needed
    def _expander(img):
        if img is not None and len(img.shape) < 3:
            img = np.expand_dims(img,-1)
        return img

    return tuple(map(_expander, images))


def remove_n_channels_dim(*images):
    # Adds a n_channels dim if needed
    def _remover(img):
        if img is not None and len(img.shape) == 3 and img.shape[2] == 1:
            img = img[:,:,0]
        return img

    return tuple(map(_remover, images))


def normalize_images(*images):
    def _normalize(img):
        if img is not None:
            img = img.astype(np.float64)[:,:,:3]
            return img
            # dont scale! it causes varying averages in different images
            # return (img - np.min(img))/(np.max(img) - np.min(img))
        return None

    # first add n_channels_dim
    images = add_n_channels_dim(*images)

    return tuple(map(_normalize, images))


# def auto_canny(image, sigma=0.33):
#     # compute the median of the single channel pixel intensities
#     image = (image * 255 if np.max(image) <= 1 else image).astype(np.uint8)
#     v = np.median(image)

#     # apply automatic Canny edge detection using the computed median
#     lower = int(max(0, (1.0 - sigma) * v))
#     upper = int(min(255, (1.0 + sigma) * v))
#     edged = Canny(image, lower, upper)

#     # return the edged image
#     return edged
