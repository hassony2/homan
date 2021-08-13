import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np


def add_clip_text(imgs, text):
    new_imgs = []
    """
    Fails for some reason :'( 
    """
    for img in imgs:
        pil_img = Image.fromarray(img)
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 50)
        draw = ImageDraw.Draw(pil_img)
        draw.text((200, img.shape[1] - 100), text, (0, 0, 0), font=font)
        new_img = np.array(pil_img)
        new_imgs.append(new_img)
    return new_imgs
