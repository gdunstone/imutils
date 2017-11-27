import cv2
import math
import numpy as np
from .templates import card_grayscale, mask_grayscale, qr_grayscale
from .transform import rotate, resize, cord_rotate
from .colour import color_correct_stats


def prep_image(img: np.ndarray) -> np.ndarray:
    """
    resizes an image to a hires one.

    :param img: image to be resized
    :type: np.ndarray
    :return: resized image, or the original image if it didnt need to be resized.
    :rtype: np.ndarray
    """
    if img.shape[0] != 3456:
        img = resize(img, height=3456, inter=cv2.INTER_CUBIC)
    return img


def _detect_template(img: np.array, template: np.array,
                     search_scale: tuple = (0.90, 1.1),
                     search_degree: tuple = (-2.5, 2.5)) -> dict:
    """
    function to detect a card within an image.
    Assumes that the card in the image is within the search scale, search degree
    within the image.

    :param img: image to detect card within.
    :type: np.ndarray
    :param template: template image, cannied and resized.
    :type: np.ndarray
    :return dict: dictionary of detection results
    """
    template_h, template_w = template.shape[:2]
    scale_search_range = np.linspace(search_scale[0], search_scale[1], 11)
    # scale_search_range = [1.1]
    degree_search_range = np.linspace(search_degree[0], search_degree[1], 10)

    # detect edges in the grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 40, 50)
    # search for the best scale and rotation degree

    results = dict()
    for degree in degree_search_range:
        gray_rot = edged if degree == 0.0 else rotate(edged, degree)

        for scale in scale_search_range:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing

            resized = resize(gray_rot, width=int(blurred.shape[1] * scale))

            ratio = gray_rot.shape[1] / resized.shape[1]
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < template_h or resized.shape[1] < template_w:
                break
            # apply template matching to find the template in the image
            method = cv2.TM_CCOEFF_NORMED
            # method = cv2.TM_CCOEFF
            result = cv2.matchTemplate(resized, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            # if we have results a new maximum correlation value, then update
            # the bookkeeping variable
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            min_check = min_val < results.get('min_val', float("Infinity"))
            topleftloc = max_loc
            check = max_val > results.get('max_val', float("-Infinity"))
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                topleftloc = min_loc
                check = min_check

            if check:
                def scale_2_resized(a):
                    return int(a * ratio)

                def tl_scale(a):
                    return scale_2_resized(a)

                def br_scale(a, s):
                    return scale_2_resized(a + s)

                topleft = tuple(map(tl_scale, topleftloc))
                botright = tuple(map(br_scale, topleftloc, (template_w, template_h)))

                results["max_val"] = max_val
                results["min_val"] = min_val
                results["max_loc"] = max_loc
                results["min_loc"] = min_loc
                results["topleft"] = topleft
                results["botright"] = botright
                results["ratio"] = ratio
                results["degree"] = degree
                results["scale"] = scale

    if len(results) is 0:
        print("Template not found.")
        return dict()

    x1, y1 = results['topleft']
    x2, y2 = results['botright']
    deg = results['degree']
    output_img = rotate(img, deg) if deg != 0 else img
    output_img = output_img[y1:y2, x1:x2, :]
    results['detected'] = output_img

    colour_threshold = 120  # a value between 0 and 255
    b, g, r = cv2.split(output_img)
    resupdate = {
        "colour_r_sum": np.sum(r > colour_threshold),
        "colour_g_sum": np.sum(g > colour_threshold),
        "colour_b_sum": np.sum(b > colour_threshold)
    }
    resupdate['colour_sum'] = sum(resupdate.values()) / (output_img.shape[0] * output_img.shape[1])
    results.update(resupdate)
    return results


def _detect_card_auto(path: str, probable_card_pos: tuple = tuple(), image_array: np.array = None) -> np.array:
    card = card_grayscale.copy()
    default_card_h = 100
    height_card, width_card = card.shape[:2]
    image_scale = default_card_h / float(height_card)
    card = cv2.resize(card, (0, 0), fx=image_scale, fy=image_scale)
    card = cv2.Canny(card, 40, 50)
    card_h, card_w = card.shape[:2]

    image = image_array
    if image_array is None:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ds = 1
    if image.shape[1] != 5184:
        ds = 5184 / image.shape[1]

    image_scale *= ds

    print("Scaling for match: {}".format(image_scale))

    image_resized = cv2.resize(image, (0, 0), fx=image_scale, fy=image_scale)

    scale_search_range = np.linspace(0.90, 1.1, 11)
    degree_search_range = np.linspace(-2.5, 2.5, 11)
    print("Detecting colour card for {}".format(path))

    # detect edges in the grayscale image
    image_resized = cv2.GaussianBlur(image_resized, (5, 5), 0)
    edged = cv2.Canny(image_resized, 40, 50)
    # search for the best scale and rotation degree

    results = dict()
    for degree in degree_search_range:
        gray_rot = edged if degree == 0.0 else rotate(edged, degree)

        for scale in scale_search_range:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing

            resized = resize(gray_rot, width=int(image_resized.shape[1] * scale))

            ratio = gray_rot.shape[1] / resized.shape[1]
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < card_h or resized.shape[1] < card_w:
                break
            # apply template matching to find the template in the image
            method = cv2.TM_CCOEFF_NORMED
            # method = cv2.TM_CCOEFF
            result = cv2.matchTemplate(resized, card, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            # if we have results a new maximum correlation value, then update
            # the bookkeeping variable
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            min_check = min_val < results.get('min_val', float("Infinity"))
            topleftloc = max_loc
            check = max_val > results.get('max_val', float("-Infinity"))
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                topleftloc = min_loc
                check = min_check

            if check:
                def scale_2_resized(a):
                    return int(a * ratio)

                def scale_2_image(a):
                    # return int(a)
                    return int(a / image_scale)

                def tl_scale(a):
                    return scale_2_image(scale_2_resized(a))

                def br_scale(a, s):
                    return scale_2_image(scale_2_resized(a + s))

                topleft = tuple(map(tl_scale, topleftloc))
                botright = tuple(map(br_scale, topleftloc, (card_w, card_h)))

                results["max_val"] = max_val
                results["min_val"] = min_val
                results["max_loc"] = max_loc
                results["min_loc"] = min_loc
                results["topleft"] = topleft
                results["botright"] = botright
                results["ratio"] = ratio
                results["degree"] = degree
                results["scale"] = scale
                # results['img'] = cv2.rectangle(image_resized, topleft, botright, (255, 0, 0), 5)

    if len(results) is 0:
        print("Colour checker not found.")
        return

    from pprint import pformat
    print("Detection complete, results: {}".format(pformat(results)))
    # outpath = "".join(path.split(".")[:-1] + ["-colourcard", os.path.splitext(path)[-1]])
    outpath = "output.jpg"
    x1, y1 = results['topleft']
    x2, y2 = results['botright']

    image = image_array
    if image_array is not None:
        import random
        rfn = hex(random.randint(16, 65)).split("x")[-1]
        # outpath = "/volumes/sites-storage/www/data/outputs/{}.jpg".format(rfn)
        # outpathweb = outpath.replace("/volumes/sites-storage/www/data", "https://data.traitcapture.org")
        outpath = "/home/stormaes/Work/site-phenocam-org/www/static/img/{}.jpg".format(rfn)
        outpathweb = outpath.replace("/home/stormaes/Work/site-phenocam-org/www/static/img",
                                     "http://localhost:5000/static/img")
        results['webpath'] = outpathweb
    else:
        image = cv2.imread(path, cv2.IMREAD_ANYCOLOR)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(outpath, image)
    return results


def colour_card_stats(img: np.ndarray) -> dict:
    """
    Detects the colour card and provides statistics on it.

    :param img: input image with a card in it (maybe)
    :type: np.ndarray
    :return: dictionary of statistics on the image
    :rtype: dict
    """
    rect_margin = int(3 * img.shape[0] / 720)

    card = card_grayscale.copy()
    default_card_h = 100
    height_card, width_card = card.shape[:2]
    image_scale = default_card_h / float(height_card)
    card = cv2.resize(card, (0, 0), fx=image_scale, fy=image_scale)
    card = cv2.Canny(card, 40, 50)

    # ds starts as 1.0
    dynamic_scaling = 1.0
    if img.shape[1] != 5184:
        # set dynamic scaling to a ratio of target/current
        dynamic_scaling = 5184 / img.shape[1]
    # apply the dynamic scaling to our values
    image_scale *= dynamic_scaling

    # Cropping Image to area where color card is supposed to be
    crop_s_x = img.shape[1] / 4
    crop_e_x = 3 * img.shape[1] / 4
    crop_s_y = img.shape[0] / 4
    crop_e_y = 3 * img.shape[0] / 4
    img_cropped = img[int(crop_s_y):int(crop_e_y), int(crop_s_x):int(crop_e_x), :]
    img_cropped_center = [img_cropped.shape[1] / 2, img_cropped.shape[0] / 2]

    # resize image.
    img_cropped = cv2.resize(img_cropped, (0, 0), fx=image_scale, fy=image_scale, interpolation=cv2.INTER_CUBIC)

    # Detecting color card and filling the relevant output fields
    results = _detect_template(img_cropped, card)
    if not len(results):
        print("template not found? wtf")
        return dict()

    acc = results.pop('max_val')
    deg = results.pop('degree')

    def scale_2_image(a):
        # very important to untransform the coordinates.
        return int(a / image_scale)

    x1, y1 = map(scale_2_image, results.pop('topleft'))
    x2, y2 = map(scale_2_image, results.pop('botright'))

    # Mapping back X and Y coordinates of detected card to the original image coordinates
    x1_rotated, y1_rotated = cord_rotate(img_cropped_center, [x1, y1], deg * math.pi / 180.0)
    x1_offset, y1_offset = x1_rotated + crop_s_x, y1_rotated + crop_s_y

    # rotate coordinates
    x2_rotated, y2_rotated = cord_rotate(img_cropped_center, [x2, y2], deg * math.pi / 180.0)
    x2_offset, y2_offset = x2_rotated + crop_s_x, y2_rotated + crop_s_y

    # Checking card and filling the relevant output fields
    upright, damaged_or_blocked, colour_correction_error = color_correct_stats(results['detected'], acc)

    results['accuracy'] = acc
    results['rotation_degrees'] = deg
    results['upright'] = not upright
    results['damaged_or_blocked'] = damaged_or_blocked
    results['region'] = [(x1_offset, y1_offset), (x2_offset, y2_offset)]
    results['colour_correction_error'] = colour_correction_error
    # Drawing a box around the detected card: green if it is all good, red otherwise
    dx1, dy1, dx2, dy2 = tuple(
        map(int, (x1_offset - rect_margin, y1_offset - rect_margin, x2_offset + rect_margin, y2_offset + rect_margin)))

    rect_colour = (0, 255, 0)

    # if damaged_or_blocked or not orientation or acc < 0.3:
    if damaged_or_blocked or acc < 0.3:
        rect_colour = (0, 0, 255)
    results['rects'] = list()
    results['rects'].append(dict(tl=(dx1, dy1), br=(dx2, dy2), colour=rect_colour))
    return results


def qr_stats(img: np.ndarray, upright=True) -> dict:
    """
    detects qr codes in an image and returns metadata about them.

    :param img: image to detect qr codes in.
    :param upright: whether the image is upright or upside down.
    :return: dictionary of values
    """
    output = dict()
    output['rects'] = list()
    output['qr_codes'] = list()

    rect_margin = int(2 * img.shape[0] / 720)
    mask = mask_grayscale.copy()
    if not upright:
        mask = rotate(mask, angle=180)

    image_scale = mask.shape[1] / img.shape[1]

    # Reading QR template and detecting the edges
    qr_template = qr_grayscale.copy()
    qr_template = cv2.Canny(qr_template, 40, 50)
    # mask resize, thresh, analysis.
    ret, mask = cv2.threshold(mask, 10, 1, cv2.THRESH_BINARY)
    mask_analyzed = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    # dynamic scaling

    # resize image.
    img = resize(img, height=mask.shape[0], inter=cv2.INTER_CUBIC)

    try:
        from pyzbar.pyzbar import decode
    except Exception as e:
        print(str(e))

    for i in [1, 2, 3, 4, 5, 6, 7, 8]:  # 8 spots where QR codes are supposed to be
        # Cropping image to location of each indivdual QR code
        try:
            mask_temp = np.where(mask_analyzed[1] == i)
            crop_s_x = min(mask_temp[1])
            crop_e_x = max(mask_temp[1])
            crop_s_y = min(mask_temp[0])
            crop_e_y = max(mask_temp[0])
            image_masked = img[crop_s_y:crop_e_y, crop_s_x:crop_e_x, :]

            img_masked_center = [image_masked.shape[1] / 2, image_masked.shape[0] / 2]
        except Exception as e:
            print(str(e))
            continue

        # Detecting QR code
        qr_results = _detect_template(image_masked, qr_template)

        if not len(qr_results):
            continue
        detected_qr_img = qr_results['detected']
        decoded_qr_value = None
        try:
            decoded_qr_value = decode(cv2.cvtColor(detected_qr_img, cv2.COLOR_BGR2GRAY))
            if type(decoded_qr_value) is list and len(decoded_qr_value) >= 1:
                decoded_qr_value = decoded_qr_value[-1]
            if decoded_qr_value and type(decoded_qr_value) is str:
                qr_results['decoded'] = decoded_qr_value
        except Exception as e:
            print(str(e))

        acc = qr_results.pop('max_val')
        scale = qr_results['scale']
        deg = qr_results.pop('degree')
        colour_sum = qr_results['colour_sum']

        def scale_2_image(a):
            # very important to untransform the coordinates.
            return int(a / image_scale)

        x1, y1 = qr_results['topleft']
        x2, y2 = qr_results['botright']
        deg_radians = deg * math.pi / float(180)

        # Mapping back X and Y coordinates of detected card to the original image coordinates
        x1_rotated, y1_rotated = cord_rotate(img_masked_center, [x1, y1], deg_radians)
        x1_offset, y1_offset = x1_rotated + crop_s_x, y1_rotated + crop_s_y

        x2_rotated, y2_rotated = cord_rotate(img_masked_center, [x2, y2], deg_radians)
        x2_offset, y2_offset = x2_rotated + crop_s_x, y2_rotated + crop_s_y

        # rescale back down to image coordinate scale.
        x1_offset, y1_offset, x2_offset, y2_offset = map(scale_2_image, (x1_offset, y1_offset, x2_offset, y2_offset))

        # drawing values:
        dx1, dy1, dx2, dy2 = tuple(map(int, (
            x1_offset - rect_margin, y1_offset - rect_margin, x2_offset + rect_margin, y2_offset + rect_margin)))
        # blue
        rect_colour = (255, 0, 0)
        # if totsum < 2.5 or acc < 0.2:
        if colour_sum < 2.5 or acc < 0.2:
            # red if not good match
            rect_colour = (0, 0, 255)
        elif decoded_qr_value:
            # green
            rect_colour = (0, 255, 0)

        qr_results['accuracy'] = acc
        qr_results['rotation_degrees'] = deg
        qr_results['upright'] = not upright
        qr_results['region'] = [(x1_offset, y1_offset), (x2_offset, y2_offset)]

        output['rects'].append(dict(tl=(dx1, dy1), br=(dx2, dy2), colour=rect_colour))
        output['qr_codes'].append(qr_results)
    return output
