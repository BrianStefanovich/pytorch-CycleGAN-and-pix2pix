from PIL import Image
import os, sys

size_images = dict()
im_size = 512
source_images_path = ""

for dirpath, _, filenames in os.walk(source_images_path):
    for path_image in filenames:
        image = os.path.abspath(os.path.join(dirpath, path_image))
        with Image.open(image) as img:
            width, heigth = img.size
            size_images[path_image] = {'width': width, 'heigth': heigth}
print(size_images)


def make_square(image, min_size=512, fill_color=(255, 255, 255, 0)):
    ''' Resize image as a square with signature in the center and black(transparent) strips at top and bottom. '''
    x, y = image.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(image, (int((size - x) / 2), int((size - y) / 2)))
    new_im = new_im.resize((im_size, im_size))
    return new_im

def resize_images(path):
    ''' Function to resize the images to the ip format for gans. '''
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path+item):
            image = Image.open(path+item)
            image = make_square(image)
            image.save(path+item)