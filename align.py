import os
import logging
import cv2
import numpy as np
from .colour import match_histogram_colour
from .transform import transform_image_from_points

logger = logging.getLogger(__name__)

def alignSURF(root, colour_template, im1_p, im2_p, transform):
    """
    aligns 2 images using SURF
    """
    # Read the images to be aligned

    im1 = cv2.imread(im1_p, cv2.IMREAD_COLOR)
    im2 = cv2.imread(im2_p, cv2.IMREAD_COLOR)

    im2 = match_histogram_colour(im2, colour_template)

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    descriptor = cv2.xfeatures2d.SURF_create(upright=True)
    logger.info("Detecting SURF")
    kp_img1, img1_desc = descriptor.detectAndCompute(im1_gray, None)
    kp_img2, img2_desc = descriptor.detectAndCompute(im2_gray, None)

    logger.info("Matching SURF")
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(img1_desc, img2_desc, k=2)
    matches = [m[0] for m in raw_matches if (len(m) == 2 and m[0].distance < m[1].distance * 0.75)]
    logger.info("Transforming SURF")
    im2_aligned = transform_image_from_points(im2, matches, kp_img1, kp_img2, transform)
    im2_fn = os.path.join(root, os.path.basename(im2_p))
    cv2.imwrite(im2_fn, im2_aligned)
    logger.info("Done SURF")
    del im1, im2, im2_aligned, im1_gray, im2_gray
    return im2_fn


def alignORB(root, colour_template, im1_p, im2_p, transform):
    # Read the images to be aligned
    cv2.ocl.setUseOpenCL(False)
    im1 = cv2.imread(im1_p, cv2.IMREAD_COLOR)
    im2 = cv2.imread(im2_p, cv2.IMREAD_COLOR)

    im2 = match_histogram_colour(im2, colour_template)

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    descriptor = cv2.ORB_create()
    logger.info("Detecting ORB")
    kp_img1, img1_desc = descriptor.detectAndCompute(im1_gray, None)
    kp_img2, img2_desc = descriptor.detectAndCompute(im2_gray, None)

    logger.info("Matching ORB")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(img1_desc, img2_desc)
    if len(matches) < 10:
        del im1, im2, im1_gray, im2_gray
        return im1_p

    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:100]

    logger.info("Transforming ORB")
    im2_aligned = transform_image_from_points(im2, matches, kp_img1, kp_img2, transform)
    im2_fn = os.path.join(root, os.path.basename(im2_p))
    cv2.imwrite(im2_fn, im2_aligned)
    logger.info("Done ORB")
    del im1, im2, im2_aligned, im1_gray, im2_gray
    return im2_fn


def alignPhase(root, colour_template, im1_p, im2_p, transform):
    # Read the images to be aligned
    im1 = cv2.imread(im1_p, cv2.IMREAD_COLOR)
    im2 = cv2.imread(im2_p, cv2.IMREAD_COLOR)

    im2 = match_histogram_colour(im2, colour_template)

    if im2.shape[0] > im1.shape[0]:
        im2 = im2[:im1.shape[0], :, :]
    else:
        im1 = im1[:im2.shape[0], :, :]

    if im2.shape[1] > im1.shape[1]:
        im2 = im2[:, :im1.shape[1], :]
    else:
        im1 = im1[:, :im2.shape[1], :]

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im1_float = np.float64(im1_gray) / 255.0
    im2_float = np.float64(im2_gray) / 255.0

    translation, some_other_var = cv2.phaseCorrelate(im1_float, im2_float)

    M = np.float32([[1, 0, -translation[0]], [0, 1, -translation[1]]])
    im2_aligned = cv2.warpAffine(im2, M, (im1.shape[1], im1.shape[0]))
    im2_fn = os.path.join(root, os.path.basename(im2_p))

    cv2.imwrite(im2_fn, im2_aligned)
    logger.info("Done Phase")
    del im1, im2, im2_aligned, im1_float, im2_float
    return im2_fn


def alignECC(root, colour_template, im1_p, im2_p, transform):
    # Read the images to be aligned
    im1 = cv2.imread(im1_p, cv2.IMREAD_COLOR)
    im2 = cv2.imread(im2_p, cv2.IMREAD_COLOR)

    im2 = match_histogram_colour(im2, colour_template)

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1_gray.shape

    # Define the motion model

    warp_mode = cv2.MOTION_HOMOGRAPHY if transform == "homo" else cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 100000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-8
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    logger.info("Calculating")
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    im2_fn = os.path.join(root, os.path.basename(im2_p))
    cv2.imwrite(im2_fn, im2_aligned)
    logger.info("ECC Done")
    del im1, im2, im2_aligned, im1_gray, im2_gray
    return im2_fn
