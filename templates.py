import cv2
import os
from .transform import resize
"""
This file provides loading of images for template matching.
"""

# this makes sure that we are loading the templates from the current directory, rather that whatever
# dir the module is being loaded from.
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

card = cv2.imread(os.path.join(__location__, "templates", "card_full_res.jpg"), cv2.IMREAD_COLOR)
card_grayscale = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

card_lores = cv2.imread(os.path.join(__location__, "templates", "card.jpg"), cv2.IMREAD_COLOR)
card_lores_grayscale = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

qr = cv2.imread(os.path.join(__location__, "templates", "QR_full_res.jpg"), cv2.IMREAD_COLOR)
qr_grayscale = cv2.cvtColor(qr, cv2.COLOR_BGR2GRAY)

qr_lores = cv2.imread(os.path.join(__location__, "templates", "QR.jpg"), cv2.IMREAD_COLOR)
qr_lores_grayscale = cv2.cvtColor(qr_lores, cv2.COLOR_BGR2GRAY)

mask = cv2.imread(os.path.join(__location__, "templates", "mask_full_res.jpg"), cv2.IMREAD_COLOR)
mask_grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

mask_lores = cv2.imread(os.path.join(__location__, "templates", "mask.jpg"), cv2.IMREAD_COLOR)
mask_lores_grayscale = cv2.cvtColor(mask_lores, cv2.COLOR_BGR2GRAY)
