from celery import Celery
from celery.utils.log import get_task_logger
from app import application
import os
from zipfile import ZipFile
import glob
import hashlib

def create_celery_app(app):
    celery = Celery(__name__, broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)

    taskbase = celery.Task

    class ContextTask(taskbase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return taskbase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask
    return celery

celery = create_celery_app(application)

logger = get_task_logger(__name__)
logger.setLevel(10)


try:
    import cv2
    import numpy as np
    cv2.ocl.setUseOpenCL(False)
except Exception as e:
    logger.error(str(e))


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Plagiarised from: http://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
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


def transform_image(img, matches, kp_ref, kp_img, transform):
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


def colour_correct(src, template):
    """
    color corrects src to template using histogram match above
    :param src: BGR image
    :param template: BGR image
    :return:
    """
    logger.info("Color correcting...")
    src = cv2.cvtColor(src, cv2.COLOR_BGR2Lab)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2Lab)
    src[:, :, 0] = hist_match(src[:, :, 0], template[:, :, 0])
    src = cv2.cvtColor(src, cv2.COLOR_Lab2BGR)
    return src


def alignSURF(colour_template, im1_p, im2_p, transform):
    # Read the images to be aligned

    im1 = cv2.imread(im1_p, cv2.IMREAD_COLOR)
    im2 = cv2.imread(im2_p, cv2.IMREAD_COLOR)

    im2 = colour_correct(im2, im1)

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    descriptor = cv2.xfeatures2d.SURF_create(upright=True)
    logger.info("Detecting SURF")
    kp_img1, img1_desc = descriptor.detectAndCompute(im1_gray, None)
    kp_img2, img2_desc = descriptor.detectAndCompute(im2_gray, None)

    logger.info("Matching SURF")
    matcher = cv2.BFMatcher()
    rawMatches = matcher.knnMatch(img1_desc, img2_desc, k=2)

    matches = [m[0] for m in rawMatches if (len(m) == 2 and m[0].distance < m[1].distance * 0.75)]
    logger.info("Transforming SURF")
    im2_aligned = transform_image(im2, matches, kp_img1, kp_img2, transform)

    im2_fn = os.path.join(os.path.dirname(im2_p), "align-SURF", os.path.basename(im2_p))
    cv2.imwrite(im2_fn, im2_aligned)
    logger.info("Done SURF")
    del im1, im2, im2_aligned, im1_gray, im2_gray
    return im2_fn


def alignORB(colour_template, im1_p, im2_p, transform):
    # Read the images to be aligned
    cv2.ocl.setUseOpenCL(False)
    im1 = cv2.imread(im1_p, cv2.IMREAD_COLOR)
    im2 = cv2.imread(im2_p, cv2.IMREAD_COLOR)

    im2 = colour_correct(im2, colour_template)

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
    im2_aligned = transform_image(im2, matches, kp_img1, kp_img2, transform)

    im2_fn = os.path.join(os.path.dirname(im2_p), "align-ORB", os.path.basename(im2_p))
    cv2.imwrite(im2_fn, im2_aligned)
    logger.info("Done ORB")
    del im1, im2, im2_aligned, im1_gray, im2_gray
    return im2_fn


def alignPhase(colour_template, im1_p, im2_p, transform):
    # Read the images to be aligned
    im1 = cv2.imread(im1_p, cv2.IMREAD_COLOR)
    im2 = cv2.imread(im2_p, cv2.IMREAD_COLOR)

    im2 = colour_correct(im2, colour_template)

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
    im1_float = np.float64(im1_gray)/255.0
    im2_float = np.float64(im2_gray)/255.0

    translation, some_other_var = cv2.phaseCorrelate(im1_float, im2_float)

    M = np.float32([[1, 0, -translation[0]], [0, 1, -translation[1]]])
    im2_aligned = cv2.warpAffine(im2, M, (im1.shape[1], im1.shape[0]))
    im2_fn = os.path.join(os.path.dirname(im2_p), "align-Phase", os.path.basename(im2_p))
    cv2.imwrite(im2_fn, im2_aligned)
    logger.info("Done Phase")
    del im1, im2, im2_aligned, im1_float, im2_float
    return im2_fn



def alignECC(colour_template, im1_p, im2_p, transform):
    # Read the images to be aligned
    im1 = cv2.imread(im1_p, cv2.IMREAD_COLOR)
    im2 = cv2.imread(im2_p, cv2.IMREAD_COLOR)

    im2 = colour_correct(im2, colour_template)

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

    im2_fn = os.path.join(os.path.dirname(im2_p), "align-ECC", os.path.basename(im2_p))
    cv2.imwrite(im2_fn, im2_aligned)
    logger.info("ECC Done")
    del im1, im2, im2_aligned, im1_gray, im2_gray
    return im2_fn


@celery.task(bind=True)
def align_images(self, images, how, transform):

    import time
    self.update_state(state='STARTED',
                      meta={'status': 'Aligning {} images'.format(str(len(images)))})
    ipth1 = images.pop(0)
    rootpath = os.path.join(os.path.dirname(ipth1), "align-ECC")

    if how == "surf":
        rootpath = os.path.join(os.path.dirname(ipth1), "align-SURF")
        meth = alignSURF
    elif how == "orb":
        rootpath = os.path.join(os.path.dirname(ipth1), "align-ORB")
        meth = alignORB
    elif how == "phase":
        rootpath = os.path.join(os.path.dirname(ipth1), "align-Phase")
        meth = alignPhase
    else:
        meth = alignECC
    os.makedirs(rootpath, exist_ok=True)

    import random
    min_size = 192038102938123, 2139071928479082374
    logger.info("Resizing...")
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
    im = cv2.imread(ipth1, cv2.IMREAD_COLOR)
    im = im[:min_size[0], :min_size[1], :]
    cv2.imwrite(ipth1, im)
    colour_template = cv2.imread(ipth1, cv2.IMREAD_COLOR)
    logger.info("Min Size: {}x{}".format(*min_size))

    for idx, ipth2 in enumerate(images):
        try:
            logger.info("{}: {}\t{}".format(meth.__name__, ipth1, ipth2))
            t = time.time()
            ipth1 = meth(colour_template, ipth1, ipth2, transform)
            logger.info("time:{0}:\t{1:.02f}".format(meth.__name__, time.time() - t))

            self.update_state(state="PROGRESS",
                              meta={'current': idx+1,
                                    'total': len(images),
                                    "status": time.time() - t})

        except Exception as e:
            logger.error(str(e))
            continue

    m = hashlib.md5()
    for a in images:
        with open(a, 'rb') as f:
            m.update(f.read())

    fn = "/volumes/sites-storage/www/data/aligned/{}.zip".format(m.hexdigest()[:6])
    logger.info("Writing zipfile to {}".format(fn))
    with ZipFile(fn, 'w') as zf:
        for fn_f in glob.glob(os.path.join(rootpath, "*")):
            zf.write(fn_f)
        logger.info("Files added. Testing zipfile.")
        bf = zf.testzip()
        while bf:
            logger.error("bad file in zip {}".format(bf))
            zf.write(bf)
            bf = zf.testzip()
        logger.info("zipfile test complete")
    url = "https://data.traitcapture.org/aligned/{}.zip".format(m.hexdigest()[:6])
    self.update_state(state="SUCCESS", meta={'url': url, 'current': 100, 'total': 100})

    # else:
    #     self.update_state(state="FAILURE",
    #                       meta={'current': 100,
    #                             'total': 100,
    #                             'status': "Couldnt align anything."})

    return {'current': 100, 'total': 100, "state": "SUCCESS", 'status': 'Task completed!', "url": url}

