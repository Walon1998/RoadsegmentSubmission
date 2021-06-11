#!/usr/bin/env python

import glob
import io
import os
from random import uniform

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from natsort import natsorted
from skimage.color import rgb2gray
from skimage.io import imsave

# This file is used to create more training data
# Stolen from here: https://github.com/daniCh8/road-segmentation-eth-cil-2020
# The above link is worth a read, they scored 2nd place last year!
# It proposes 4 masks, you can choose one by writing 1,2,3,4, if you write nothing the image is skipped.

los_angeles = [
    {'n': 34.269260, 'w': -118.604202, 's': 34.171040, 'e': -118.370722},
    {'n': 34.100406, 'w': -118.362530, 's': 33.797995, 'e': -117.863483},
    {'n': 33.714559, 'w': -118.033473, 's': 33.636157, 'e': -117.746060}
]

chicago = [
    {'n': 42.072123, 'w': -88.311501, 's': 41.643560, 'e': -87.682533}
]

houston = [
    {'n': 29.875249, 'w': -95.563377, 's': 29.610542, 'e': -95.189842}
]

phoenix = [
    {'n': 33.688554, 'w': -112.381892, 's': 33.392095, 'e': -111.887507}
]

philadelphia = [
    {'n': 40.052889, 'w': -75.233393, 's': 39.904511, 'e': -75.140009},
    {'n': 40.049736, 'w': -75.144129, 's': 40.026079, 'e': -75.027399}
]

san_francisco = [
    {'n': 37.801910, 'w': -122.506267, 's': 37.737590, 'e': -122.398120},
    {'n': 37.826862, 'w': -122.295123, 's': 37.800282, 'e': -122.255984}
]

boston = [
    {'n': 42.387338, 'w': -71.141267, 's': 42.283792, 'e': -71.046510}
]


# Save images and mask in the appropriate location
def save_image(image, mask):
    img_path = "./Data/Additional_Data/Images/{}.png".format(counter)
    mask_path = "./Data/Additional_Data/Masks/{}.png".format(counter)
    imsave(img_path, image)
    imsave(mask_path, (mask * 255).astype(np.uint8))
    print("saved image at path: {}".format(img_path))
    print("saved mask at path: {}".format(mask_path))


def pick_random_image_from_city(x, y, zoom_=18, width_=600, height_=600):
    url = "https://maps.googleapis.com/maps/api/staticmap?"
    center = "center=" + str(x) + "," + str(y)
    zoom = "&zoom=" + str(zoom_)
    size = "&size=" + str(width_) + "x" + str(height_)
    sat_maptype = "&maptype=satellite"
    road_maptype = "&maptype=roadmap"
    no_banners = "&style=feature:all|element:labels|visibility:off"
    api_key = "&key=" + "AIzaSyDUFtCvsoSpMSb0rAPR3qXfS1kACAhf1lc"  # this is my api key, you can use it for this project.

    sat_url = url + center + zoom + size + sat_maptype + no_banners + api_key
    road_url = url + center + zoom + size + road_maptype + no_banners + api_key

    sat_tmp = Image.open(io.BytesIO(requests.get(sat_url).content))
    road_tmp = Image.open(io.BytesIO(requests.get(road_url).content))
    sat_image = np.array(sat_tmp.convert('RGB'))
    roadmap = np.array(road_tmp.convert('RGB'))
    mask = np.floor(rgb2gray(np.floor(roadmap / 255))).astype(np.float32)
    new_mask = np.floor(rgb2gray(np.floor(roadmap >= 254))).astype(np.float32)
    third_mask_a = (roadmap[:, :, 0] == 255) & (roadmap[:, :, 1] == 235) & (roadmap[:, :, 2] == 161)
    third_mask_b = (roadmap[:, :, 0] == 255) & (roadmap[:, :, 1] == 242) & (roadmap[:, :, 2] == 175)
    third_mask = (third_mask_a | third_mask_b | new_mask.astype(np.bool)).astype(np.float32)
    fourth_mask = (third_mask_a | third_mask_b | mask.astype(np.bool)).astype(np.float32)

    return sat_image, mask, new_mask, third_mask, fourth_mask, roadmap


def plot_masks(image, mask, new_mask, third_mask, fourth_mask, roadmap):
    plt.figure(figsize=(15, 7))

    ax1 = plt.subplot2grid((2, 4), (0, 1))
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('image')

    ax2 = plt.subplot2grid((2, 4), (1, 0))
    ax2.imshow(mask, cmap='Greys_r')
    ax2.axis('off')
    ax2.set_title('mask 1')

    ax3 = plt.subplot2grid((2, 4), (1, 1))
    ax3.imshow(new_mask, cmap='Greys_r')
    ax3.axis('off')
    ax3.set_title('mask 2')

    ax4 = plt.subplot2grid((2, 4), (1, 2))
    ax4.imshow(third_mask, cmap='Greys_r')
    ax4.axis('off')
    ax4.set_title('mask 3')

    ax5 = plt.subplot2grid((2, 4), (1, 3))
    ax5.imshow(fourth_mask, cmap='Greys_r')
    ax5.axis('off')
    ax5.set_title('mask 4')

    ax6 = plt.subplot2grid((2, 4), (0, 2))
    ax6.imshow(roadmap)
    ax6.axis('off')
    ax6.set_title('roadmap')

    plt.show()

    print('')


def get_center(image):
    xx = int(image.shape[0] / 2)
    yy = int(image.shape[1] / 2)
    return image[xx - 200:xx + 200, yy - 200:yy + 200]


# The cities from which we choose images
cities_boxes = [los_angeles, chicago, houston, phoenix, philadelphia, san_francisco, boston]

# Find out how many images there already are such that we dont overwrite any.
temp = natsorted(glob.glob('Data/Additional_Data/Images/*'))[-1]

counter = int(os.path.basename(temp).split(".")[0]) + 1
plt.close('all')

# propose 100 newly generate images, be aware sometimes they are trash!
for i in range(100):

    city_nr = np.random.randint(len(cities_boxes))  # pick a city
    index = np.random.randint(len(cities_boxes[city_nr]))
    box = cities_boxes[city_nr][index]

    rand_x = uniform(box['w'], box['e'])
    rand_y = uniform(box['n'], box['s'])

    image, mask, new_mask, third_mask, fourth_mask, roadmap = pick_random_image_from_city(rand_y, rand_x)  # visually pick the most appropriate mask for the given map
    image = get_center(image)
    mask = get_center(mask)
    new_mask = get_center(new_mask)
    third_mask = get_center(third_mask)
    fourth_mask = get_center(fourth_mask)
    roadmap = get_center(roadmap)

    plot_masks(image, mask, new_mask, third_mask, fourth_mask, roadmap)

    text = input()

    if text == "1":
        save_image(image, mask)
    elif text == "2":
        save_image(image, new_mask)

    elif text == "3":
        save_image(image, third_mask)

    elif text == "4":
        save_image(image, fourth_mask)

    plt.clf()
    counter = counter + 1

plt.close('all')
