import cv2
import math
import numpy as np


def crop_image_to_quarter(image_orig: np.array, coordinates: tuple) -> np.array:
    """
    crops the given image to a quarter of the dimensions of the image around the point`coordinates

    :param image_orig: image to crop
    :type: np.ndarray
    :param coordinates: coordinates to crop around
    :type: tuple[int,int]
    :return: cropped image
    :rtype: np.ndarray
    """
    img_h, img_w = image_orig.shape[:2]
    x_coord, y_coord = coordinates

    x_start = max(0, x_coord - int(img_w / 4))
    y_start = max(0, y_coord - int(img_h / 4))

    x_end = min(img_w - 1, x_coord + int(img_w / 4))
    y_end = min(img_h - 1, y_coord + int(img_h / 4))

    if len(image_orig.shape) == 2:
        image_cropped = image_orig[int(y_start):int(y_end), int(x_start):int(x_end)]
    else:
        # handle images that have more than one channel
        image_cropped = image_orig[int(y_start):int(y_end), int(x_start):int(x_end), :]
    return image_cropped


def cord_rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.

    :param origin: x,y origin
    :type: tuple[float, float]
    :param point: x,y point to rotate around the origin
    :type: tuple[float, float]
    :param angle: angle to rotate in radians
    :return: rotated point
    :rtype: tuple[int, int]
    """

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)


def translate(image, x, y):
    """
    translatest an image by (x, y) pixels
    :param image:
    :param x: horizontal translation
    :param y: vertical translation
    :return:
    """
    # Define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


def rotate(image: np.array, angle: float, center: tuple = None, scale: float = 1.0) -> np.array:
    """
    rorates the image around center, or the center of the image if no center is specified
    :param image:
    :param angle: angle of rotation
    :param center: optional 2 length tuple of the center of rotation
    :param scale: optional scale
    :return:
    """
    # Grab the dimensions of the image
    (h, w) = image.shape[:2]

    # If the center is None, initialize it as the center of
    # the image
    center = center or (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (w, h))


def resize(image: np.array, width: float = None, height: float = None, inter: int = cv2.INTER_AREA) -> np.array:
    """
    resizes an image to a target width or height, with width used if both are provided

    :param image: image to resize
    :type: np.ndarray
    :param width: desired width in pixels
    :type: int
    :param height: desired height in pixels
    :type: int
    :param inter: interpolation method to use.
    :type: int, cv2 interpolation enum)
    :return: the resized image
    :rtype: np.ndarray
    """
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # grab the image size
    (h, w) = image.shape[:2]
    r = (width / float(w)) if width else (height / float(h))
    dim = (width, int(h * r)) if width else (int(w * r), height)
    return cv2.resize(image, dim, interpolation=inter)


def massacre_image_dimensions(images: list):
    """
    Makes all images in the list match the dimensions of the smallest image, and writes the images back out.
    tries to conserve memory by only reading the images in grayscale to determine the width/height


    :param images: list of image paths
    :type: list(str)
    """
    min_size = 1920381029381233423, 2139071928479082374

    for p in images:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im.shape[0] < min_size[0]:
            min_size = im.shape[0], min_size[1]
        if im.shape[1] < min_size[1]:
            min_size = min_size[0], im.shape[1]

    for p in images:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        im = im[:min_size[0], :min_size[1], :]
        cv2.imwrite(p, im)


def transform_image_from_points(img: np.array, matches: list, kp_ref: list, kp_img: list, transform: str = "rigid"):
    """
    transfroms the image `img` using a list of keypoints from a source and target image.
    uses the optional parameter "transform" which if set to "homo", will use a homography transform other than a rigid
    one.

    :param img: image to transform
    :type: np.array
    :param matches: list of matches
    :type: list(cv2.Match)
    :param kp_ref: list of keypoints for the reference points
    :param kp_img: list of keypoints for the target image points
    :param transform: how to transform the image
    :return: transformed image
    :rtype: np.array
    """
    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_img[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    if transform == "homo":
        # homography transform, better
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20.0)
        return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    else:
        # rigid transform,
        M = cv2.estimateRigidTransform(src_pts, dst_pts, False)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
