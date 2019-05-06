import numpy as np
import os
from PIL import Image, ImageOps
import imageio

def get_resized_image(img_path, width, height, save=True):
    image = Image.open(img_path)
    # PIL is column major so you have to swap the places of width and height
    image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)
    if save:
        image_dirs = img_path.split('/')
        image_dirs[-1] = 'resized_' + image_dirs[-1]
        out_path = '/'.join(image_dirs)
        if not os.path.exists(out_path):
            image.save(out_path)
    image = np.asarray(image, np.float32)
    return np.expand_dims(image, 0)

def init_random_image(content_image, width, height, noise_ratio=0.6):
    noise_image = np.random.uniform(-20, 20, (1, height, width, 3)).astype(np.float32)
    return noise_image * noise_ratio + content_image * (1 - noise_ratio)

def save_image(path, image):
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    imageio.imwrite(path, image)



