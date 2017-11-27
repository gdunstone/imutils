import numpy as np
import cv2
from scipy import optimize
from .CONSTANTS import ColorCheckerRGB_CameraTrax, ColorCheckerRGB_XRite


def _histogram_colour_match(source: np.array, template: np.array):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Plagiarised from: http://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x

    :param source: Image to transform; the histogram is computed over the flattened array
    :type: np.ndarray
    :param template: Template image; can have different dimensions to source
    :type: np.ndarray
    :return: the transformed output image
    :rtype: np.ndarray
    """
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source,
                                            return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def match_histogram_colour(src, template):
    """
    color corrects src to template using histogram match above

    :param src: BGR image
    :param template: BGR image
    :return:
    """
    src = cv2.cvtColor(src, cv2.COLOR_BGR2Lab)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2Lab)
    src[:, :, 0] = _histogram_colour_match(src[:, :, 0], template[:, :, 0])
    src = cv2.cvtColor(src, cv2.COLOR_Lab2BGR)
    return src


def _classic_gamma_correction_model(colors, color_alpha, color_constant,
                                    color_gamma):
    """Apply color correction to a list of colors.
    This uses classic gamma correction algorithm:
       |R_out|   |alpha_R    0       0   |   |R_in|^|gamma_R|   |beta_R|
       |G_out| = |   0    alpha_G    0   | * |G_in| |gamma_G| + |beta_G|
       |B_out|   |   0       0    alpha_B|   |B_in| |gamma_B|   |beta_B|

    """
    assert (colors.shape[0] == 3)
    assert (color_alpha.size == 3)
    assert (color_constant.size == 3)
    assert (color_gamma.size == 3)

    corrected_colors = np.zeros_like(colors)
    for j in range(3):
        corrected_colors[j, :] = \
            color_alpha[j] * np.power(colors[j, :], color_gamma[j]) + \
            color_constant[j]
    return corrected_colors


def _gamma_correction_model(colors, color_alpha, color_constant,
                            color_gamma):
    """
    Apply color correction to a list of colors.
    This uses a modified gamma correction algorithm:
       |R_out'|   |alpha_11 alpha_12 alpha_13|   |R_in|   |beta_R|
       |G_out'| = |alpha_21 alpha_22 alpha_23| * |G_in| + |beta_G|
       |B_out'|   |alpha_31 alpha_32 alpha_33|   |B_in|   |beta_B|

       |R_out|         |R_out'/255|^|gamma_R|
       |G_out| = 255 * |G_out'/255| |gamma_G|
       |B_out|         |B_out'/255| |gamma_B|

    """
    assert (colors.shape[0] == 3)
    assert (color_alpha.shape == (3, 3))
    assert (color_constant.size == 3)
    assert (color_gamma.size == 3)

    scaled_colors = np.dot(color_alpha, colors) + color_constant
    np.clip(scaled_colors, 0, None,
            scaled_colors)  # set min values to zeros # I (MEF) commented this. This is now like the pipeline!!
    corrected_colors = np.zeros_like(scaled_colors)
    for j in range(3):
        corrected_colors[j, :] = 255.0 * np.power(scaled_colors[j, :] / 255.0,
                                                  color_gamma[j])
    return corrected_colors


def _get_color_error(colour_alpha_constant_gamma, true_colors, actual_colors, algorithm):
    """
    Calculated color error after applying color correction.
    This function is used in :func:`get_color_correction_parameters`

    :param colour_alpha_constant_gamma:
    :param true_colors:
    :param actual_colors:
    :param algorithm:
    :return:
    """

    if algorithm == "classic_gamma_correction":
        color_alpha = colour_alpha_constant_gamma[:3].reshape([3, 1])
        color_constant = colour_alpha_constant_gamma[3:6].reshape([3, 1])
        # forced non-negative exponential component
        color_gamma = np.abs(colour_alpha_constant_gamma[6:9].reshape([3, 1]))
        corrected_colors = _classic_gamma_correction_model(actual_colors, color_alpha,
                                                           color_constant, color_gamma)
    elif algorithm == "gamma_correction":
        color_alpha = colour_alpha_constant_gamma[:9].reshape([3, 3])
        color_constant = colour_alpha_constant_gamma[9:12].reshape([3, 1])
        # forced non-negative exponential component
        color_gamma = np.abs(colour_alpha_constant_gamma[12:15].reshape([3, 1]))
        corrected_colors = _gamma_correction_model(actual_colors, color_alpha,
                                                   color_constant, color_gamma)
    else:
        raise ValueError("Unsupported algorithm {}.".format(algorithm))

    diff_colors = true_colors - corrected_colors
    # TODO: URGENT
    # the following line causes a multiplication overflow, also sqrt sucks balls.
    errors = np.sqrt(np.sum(diff_colors * diff_colors, axis=0)).tolist()
    return errors


def color_correct_stats(card, Acc):
    """
    Gets the accuracy stats for colour correction

    :param card:
    :param Acc:
    :return:
    """
    CardRGB = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
    actual_colors, actual_colors_std = get_colorcard_colors(CardRGB, grid_size=[6, 4])
    cnt_color = 0
    card_orientation = True
    card_damaged = False
    if np.sum(actual_colors[:, 8]) > np.sum(actual_colors[:, -9]):
        cnt_color = cnt_color + 1
    if np.sum(actual_colors[:, 5]) > np.sum(actual_colors[:, -6]):
        cnt_color = cnt_color + 1
    if np.sum(actual_colors[:, 0]) < np.sum(actual_colors[:, -1]):
        cnt_color = cnt_color + 1
    if cnt_color >= 2:
        actual_colors = actual_colors[:, ::-1]
        actual_colors_std = actual_colors_std[::-1]
        print('   detected card is rotated')
        card_orientation = False

    true_colors = ColorCheckerRGB_CameraTrax
    Check = True
    if any(actual_colors_std > 40):
        print('   Some colors on the colorcard seem corrupted :(')
        card_damaged = True
    actual_colors2 = actual_colors
    iter = 0
    while Check:
        iter = iter + 1
        color_alpha, color_constant, color_gamma = _get_color_correction_parameters(true_colors, actual_colors2,
                                                                                    'gamma_correction')
        corrected_colors = _gamma_correction_model(actual_colors2, color_alpha, color_constant, color_gamma)
        diff_colors = true_colors - corrected_colors
        errors = np.sqrt(np.sum(diff_colors * diff_colors, axis=0)).tolist()
        error_mean = np.mean(errors)
        if Acc > 0.4 and error_mean > 40 and iter < 3:
            actual_colors2 = actual_colors + np.random.rand(3, 24)
            print('Corrction error high, checking again....!')
        else:
            Check = False

    if error_mean > 50:  # equivalent to 20% error
        print('Image correction error out of range, {}!'.format(error_mean))

    corection_error = round((np.mean(errors) / 255) * 10000) / float(100)
    return card_orientation, card_damaged, corection_error


def _get_color_correction_parameters(true_colors, actual_colors, algorithm="gamma_correction"):
    """
    Estimate parameters of color correction function.

    Parameters
    ----------
    true_colors : 3xN ndarray
        The input ground-truth colors.
    actual_colors : 3xN ndarray
        The input actual color as captured in image.
    algorithm : string
        The correction algorithm, either `classic_gamma_correction` or
        `gamma_correction` (default)

    Returns
    -------
    color_alpha : ndarray
        The scaling coefficient.
    color_constant : ndarray
        The color constant component.
    color_gamma : ndarray
        The gamma coefficient or the exponential component of
        correction function.

    Raises
    ------
    ValueError
        If the input algorithm is not supported.
    """
    if algorithm == "classic_gamma_correction":
        color_alpha = np.ones([3, 1])
    elif algorithm == "gamma_correction":
        color_alpha = np.eye(3)
    else:
        raise ValueError("Unsupported algorithm {}.".format(algorithm))

    color_constant = np.zeros([3, 1])
    color_gamma = np.ones([3, 1])

    args_init = np.concatenate((color_alpha.reshape([color_alpha.size]),
                                color_constant.reshape([color_constant.size]),
                                color_gamma.reshape([color_gamma.size])))
    args_refined, _ = optimize.leastsq(_get_color_error, args_init,
                                       args=(true_colors, actual_colors, algorithm),
                                       maxfev=20000)

    if algorithm == "classic_gamma_correction":
        color_alpha = args_refined[:3].reshape([3, 1])
        color_constant = args_refined[3:6].reshape([3, 1])
        # forced non-negative exponential compnent
        color_gamma = np.abs(args_refined[6:9].reshape([3, 1]))
    elif algorithm == "gamma_correction":
        color_alpha = args_refined[:9].reshape([3, 3])
        color_constant = args_refined[9:12].reshape([3, 1])
        # forced non-negative exponential compnent
        color_gamma = np.abs(args_refined[12:15].reshape([3, 1]))
    else:
        raise ValueError("Unsupported algorithm {}.".format(algorithm))

    return color_alpha, color_constant, color_gamma


def get_colorcard_colors(color_card, grid_size):
    """
    Extract color information from a cropped image of a color card.
    containing squares of different colors.

    Parameters
    ----------
    color_card : ndarray
        The input cropped image containing only color card.
    grid_size : list, [horizontal_grid_size, vertical_grid_size]
        The number of columns and rows in color card.

    Returns
    -------
    colors : 3xN ndarray
        List of colors with color channels go along the first array axis.
    """
    grid_cols, grid_rows = grid_size
    colors = np.zeros([3, grid_rows * grid_cols])
    colors_std = np.zeros(grid_rows * grid_cols)

    sample_size_row = int(0.2 * color_card.shape[0] / grid_rows)
    sample_size_col = int(0.2 * color_card.shape[1] / grid_cols)
    for row in range(grid_rows):
        for col in range(grid_cols):
            r = int((row + 0.5) * color_card.shape[0] / grid_rows)
            c = int((col + 0.5) * color_card.shape[1] / grid_cols)
            i = row * grid_cols + col
            for j in range(colors.shape[0]):
                channel = color_card[r - sample_size_row:r + sample_size_row,
                          c - sample_size_col:c + sample_size_col,
                          j]
                colors[j, i] = np.median(channel.astype(np.float))
                colors_std[i] = colors_std[i] + np.std(channel.astype(np.float))

    return colors, colors_std
