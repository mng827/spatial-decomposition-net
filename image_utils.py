import cv2
import numpy as np
import scipy.ndimage.interpolation
import skimage.transform
import nibabel as nib


def to_one_hot(array, depth):
    array_reshaped = np.reshape(array, -1).astype(np.uint8)
    array_one_hot = np.zeros((array_reshaped.shape[0], depth))
    array_one_hot[np.arange(array_reshaped.shape[0]), array_reshaped] = 1
    array_one_hot = np.reshape(array_one_hot, array.shape + (-1,))

    return array_one_hot

# Copied from https://github.com/baiwenjia/ukbb_cardiac/blob/master/common/image_utils.py
def crop_image(image, cx, cy, size, constant_values=0):
    """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
    X, Y = image.shape[:2]
    rX = int(size[0] / 2)
    rY = int(size[1] / 2)
    x1, x2 = cx - rX, cx + rX
    y1, y2 = cy - rY, cy + rY
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    # Crop the image
    crop = image[x1_: x2_, y1_: y2_]
    # Pad the image if the specified size is larger than the input image size
    if crop.ndim == 3:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)),
                      'constant', constant_values=constant_values)
    elif crop.ndim == 4:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)),
                      'constant', constant_values=constant_values)
    else:
        print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
        exit(0)
    return crop, x1, y1


def resize_image(image, size, interpolation_order):
    return skimage.transform.resize(image, tuple(size), order=interpolation_order, mode='constant')


def augment_data_2d(whole_image, whole_label, preserve_across_slices, max_shift=10, max_rotate=10, max_scale=0.1, max_intensity=0.1):
    new_whole_image = np.zeros_like(whole_image)

    if whole_label is not None:
        new_whole_label = np.zeros_like(whole_label)
    else:
        new_whole_label = None

    for i in range(whole_image.shape[-1]):
        image = whole_image[:, :, i]
        new_image = image

        # For each image slice, generate random affine transformation parameters
        # using the Gaussian distribution
        if preserve_across_slices and i is not 0:
            pass
        else:
            shift_val = [np.clip(np.random.normal(), -3, 3) * max_shift,
                         np.clip(np.random.normal(), -3, 3) * max_shift]
            rotate_val = np.clip(np.random.normal(), -3, 3) * max_rotate
            scale_val = 1 + np.clip(np.random.normal(), -3, 3) * max_scale
            intensity_val = 1 + np.clip(np.random.normal(), -3, 3) * max_intensity

        new_whole_image[:, :, i] = transform_data_2d(new_image, shift_val, rotate_val, scale_val, interpolation_order=1)
        new_whole_image[:, :, i] = new_whole_image[:, :, i] * intensity_val

        if whole_label is not None:
            label = whole_label[:, :, i]
            new_label = label
            new_whole_label[:, :, i] = transform_data_2d(new_label, shift_val, rotate_val, scale_val,
                                                         interpolation_order=0)

    return new_whole_image, new_whole_label


def transform_data_2d(image, shift_value, rotate_value, scale_value, interpolation_order):
    # Apply the affine transformation (rotation + scale + shift) to the image
    row, col = image.shape
    M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_value, 1.0 / scale_value)
    M[:, 2] += shift_value

    return scipy.ndimage.interpolation.affine_transform(image, M[:, :2], M[:, 2], order=interpolation_order)


def save_nii(image, affine, header, filename):
    if header is not None:
        nii_image = nib.Nifti1Image(image, None, header=header)
    else:
        nii_image = nib.Nifti1Image(image, affine)

    nib.save(nii_image, filename)
    return


def load_nii(nii_image):
    image = nib.load(nii_image)
    affine = image.header.get_best_affine()
    image = image.get_data()

    return image, affine


def np_log_likelihood(prob, truth, num_classes):
    truth_one_hot = to_one_hot(truth, depth=num_classes)
    return np.mean(np.log(np.sum(prob * truth_one_hot, axis=-1) + 1e-7))


def np_categorical_dice_3d(pred, truth, num_classes):
    return np_categorical_dice(pred, truth, num_classes, axis=(0, 1, 2))


def np_categorical_dice(pred, truth, num_classes, axis, smooth_epsilon=None):
    pred_one_hot = to_one_hot(pred, depth=num_classes)
    truth_one_hot = to_one_hot(truth, depth=num_classes)

    numerator = 2 * np.sum(pred_one_hot * truth_one_hot, axis=axis)
    denominator = np.sum(pred_one_hot, axis=axis) + np.sum(truth_one_hot, axis=axis)

    if smooth_epsilon is None:
        return numerator / (denominator + 1e-7)
    else:
        return (numerator + smooth_epsilon) / (denominator + smooth_epsilon)


def np_foreground_dice(pred, truth):
    foreground_pred = np.zeros_like(pred)
    foreground_pred[pred != 0] = 1

    foreground_gt = np.zeros_like(truth)
    foreground_gt[truth != 0] = 1

    return np_categorical_dice_3d(foreground_pred, foreground_gt, num_classes=2)[1]


def np_categorical_assd_hd(pred, truth, num_classes, pixel_spacing):
    assd = np.zeros(num_classes)
    hd = np.zeros(num_classes)

    for i in range(num_classes):
        assd[i], hd[i] = distance_metric_2d(pred == i, truth == i, pixel_spacing, average_slices=True)

    return assd, hd


def np_foreground_assd_hd(pred, truth, pixel_spacing):
    foreground_pred = np.zeros_like(pred)
    foreground_pred[pred != 0] = 1

    foreground_gt = np.zeros_like(truth)
    foreground_gt[truth != 0] = 1

    return np_categorical_assd_hd(foreground_pred, foreground_gt, num_classes=2, pixel_spacing=pixel_spacing)[1]

# Modified from https://github.com/baiwenjia/ukbb_cardiac/blob/master/common/image_utils.py
def distance_metric_2d(seg_A, seg_B, pixel_spacing, average_slices, fill_nan=False):
    """
        Measure the distance errors between the contours of two segmentations.
        The manual contours are drawn on 2D slices.
        We calculate contour to contour distance for each slice.
    """
    table_md = []
    table_hd = []
    X, Y, Z = seg_A.shape
    for z in range(Z):
        # Binary mask at this slice
        slice_A = seg_A[:, :, z].astype(np.uint8)
        slice_B = seg_B[:, :, z].astype(np.uint8)

        # The distance is defined only when both contours exist on this slice
        if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
            # Find contours and retrieve all the points
            _, contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1),
                                              cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_NONE)

            pts_A = np.concatenate(contours, axis=0)[:, 0, :] * pixel_spacing

            _, contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1),
                                              cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_NONE)

            pts_B = np.concatenate(contours, axis=0)[:, 0, :] * pixel_spacing

            # Distance matrix between point sets
            N = np_pairwise_squared_euclidean_distance(pts_A, pts_B)
            N = np.sqrt(N)

            # Distance matrix between point sets
            # M = np.zeros((len(pts_A), len(pts_B)))
            # for i in range(len(pts_A)):
            #   for j in range(len(pts_B)):
            #     M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

            # print(np.allclose(M, N, rtol=1e-5, atol=1e-5))

            # Mean distance and hausdorff distance
            md = 0.5 * (np.mean(np.min(N, axis=0)) + np.mean(np.min(N, axis=1)))
            hd = np.max([np.max(np.min(N, axis=0)), np.max(np.min(N, axis=1))])
            table_md += [md]
            table_hd += [hd]
        elif fill_nan:
            if np.sum(slice_A) == 0 and np.sum(slice_B) == 0:
                table_md += [0.]
                table_hd += [0.]
            elif np.sum(slice_A) == 0:
                mean_distance = find_average_distance_within_contour(slice_B, pixel_spacing)
                table_md += [mean_distance]
                table_hd += [mean_distance]
            else:
                mean_distance = find_average_distance_within_contour(slice_A, pixel_spacing)
                table_md += [mean_distance]
                table_hd += [mean_distance]
        else:
            table_md += [np.nan]
            table_hd += [np.nan]

    if average_slices:
        # Return the mean distance and Hausdorff distance across 2D slices
        mean_md = np.nanmean(table_md) if table_md else None
        mean_hd = np.nanmean(table_hd) if table_hd else None
    else:
        mean_md = table_md
        mean_hd = table_hd

    return mean_md, mean_hd


def find_average_distance_within_contour(slice, pixel_spacing):
    if np.sum(slice) == 0:
        return 0

    _, contours, _ = cv2.findContours(cv2.inRange(slice, 1, 1), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    pts = np.concatenate(contours, axis=0)[:, 0, :] * pixel_spacing

    N = np_pairwise_squared_euclidean_distance(pts, pts)
    N = np.sqrt(N)

    return np.mean(np.max(N, axis=0))


def np_pairwise_squared_euclidean_distance(x, z):
    '''
    This function calculates the pairwise euclidean distance
    matrix between input matrix x and input matrix z and
    return the distances matrix as result.

    x is a BxN matrix
    z is a CxN matrix
    result d is a BxC matrix that contains the Euclidean distances

    '''
    # Calculate the square of both
    x_square = np.expand_dims(np.sum(np.square(x), axis=1), axis=1)
    z_square = np.expand_dims(np.sum(np.square(z), axis=1), axis=0)

    # Calculate x*z
    x_z = np.matmul(x, np.transpose(z))

    # Calculate squared Euclidean distance
    d_matrix = x_square + z_square - 2 * x_z
    d_matrix[d_matrix < 0] = 0

    return d_matrix
