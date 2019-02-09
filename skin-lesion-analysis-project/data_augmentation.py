import pandas as pd
import numpy as np
from skimage import io
from skimage.util import random_noise
from skimage.transform import rotate, resize, swirl
from skimage import exposure
import matplotlib.pyplot as plt
from tqdm import tqdm


def plt_im(image):
    plt.imshow(image)
    plt.show()


def image_to_np(row, dataframe, plot=False):
    im_name = dataframe['image'][row]
    image = io.imread("./data/ISIC2018_Training_Input/" + im_name + ".jpg")

    if plot:
        plt.imshow(image)
        plt.show()

    return np.array(image)


def brighten(image):
    return exposure.adjust_gamma(image, gamma=0.5, gain=0.9)


def sigmoid(image):
    return exposure.adjust_sigmoid(image, gain=7)


def swirl_im(image, strength=3):
    return swirl(image, mode='reflect', radius=300, strength=strength)


def contrast(image):
    v_min, v_max = np.percentile(image, (0.4, 99.6))
    better_contrast = exposure.rescale_intensity(image, in_range=(v_min, v_max))
    return better_contrast


def random_rotation(image):
    # pick a random degree of rotation between 45 deg on the left and 45 deg on the right
    random_degree = np.random.uniform(-25, 25)
    return rotate(image, random_degree, mode='symmetric')


def augment_data(class_id, image_indexes, dataframe):
    """

    :param class_id: Integer in [0-6] representing a specific class.
    :param image_indexes: List storing image names for each class.
    :param dataframe: Dataframe storing GroundTruth table (image names and labels).
    :return: Augmented class dataset.
    """
    print('cls ID: ', class_id)
    idx = image_indexes[class_id]
    if class_id == 2:
        strt = 360
    if class_id == 4:
        strt = 959

    print('strt: ', strt)

    for i in tqdm(range(strt, len(idx)), desc="Data augmentation"):

        im = image_to_np(idx[i], dataframe)
        plt.imsave('/media/oguzhan/oguz/augment_usb/cls{}/cls{}_{}_ud.jpg'.format(class_id, class_id, i), np.flipud(im))
        plt.imsave('/media/oguzhan/oguz/augment_usb/cls{}/cls{}_{}_lr.jpg'.format(class_id, class_id, i), np.fliplr(im))
        plt.imsave('/media/oguzhan/oguz/augment_usb/cls{}/cls{}_{}_rotate.jpg'.format(class_id, class_id, i), random_rotation(im))
        plt.imsave('/media/oguzhan/oguz/augment_usb/cls{}/cls{}_{}_noise.jpg'.format(class_id, class_id, i), random_noise(im))
        plt.imsave('/media/oguzhan/oguz/augment_usb/cls{}/cls{}_{}_contrast.jpg'.format(class_id, class_id, i), contrast(im))
        plt.imsave('/media/oguzhan/oguz/augment_usb/cls{}/cls{}_{}_bright.jpg'.format(class_id, class_id, i), brighten(im))
        plt.imsave('/media/oguzhan/oguz/augment_usb/cls{}/cls{}_{}_swirl.jpg'.format(class_id, class_id, i), swirl_im(im))
        plt.imsave('/media/oguzhan/oguz/augment_usb/cls{}/cls{}_{}_sigmoid.jpg'.format(class_id, class_id, i), sigmoid(im))

        plt.imsave('/media/oguzhan/oguz/augment_usb/cls{}/cls{}_{}_swirl_rot.jpg'.format(class_id, class_id, i), random_rotation(swirl_im(im, strength=2)))
        plt.imsave('/media/oguzhan/oguz/augment_usb/cls{}/cls{}_{}_swirl_rot_bri.jpg'.format(class_id, class_id, i), brighten(random_rotation(swirl_im(im, strength=2.5))))
        plt.imsave('/media/oguzhan/oguz/augment_usb/cls{}/cls{}_{}_swirl_rot_contr.jpg'.format(class_id, class_id, i), contrast(random_rotation(swirl_im(im, strength=3))))
        plt.imsave('/media/oguzhan/oguz/augment_usb/cls{}/cls{}_{}_swirl_sig.jpg'.format(class_id, class_id, i), sigmoid(swirl_im(im, strength=2)))
        plt.imsave('/media/oguzhan/oguz/augment_usb/cls{}/cls{}_{}_rot_lr.jpg'.format(class_id, class_id, i), random_rotation(np.fliplr(im)))
        plt.imsave('/media/oguzhan/oguz/augment_usb/cls{}/cls{}_{}_rot_ud.jpg'.format(class_id, class_id, i), random_rotation(np.flipud(im)))
        plt.imsave('/media/oguzhan/oguz/augment_usb/cls{}/cls{}_{}_rot_contr.jpg'.format(class_id, class_id, i), random_rotation(contrast(im)))


def separate_images_into_class_folders():
    """
    Separate images of each class and save to Balanced_Trainset folder.

    """
    for i in range(len(im_idx)):
        print('class in operation: ', i)
        for j in tqdm(range(len(im_idx[i]))):
            #if i != 1:
            img = image_to_np(im_idx[i][j], df)
            plt.imsave('./data/Balanced_Trainset/cls{}/{}.jpg'.format(i, df['image'][im_idx[i][j]]), img)


#%% MAIN

df = pd.read_csv('./data/ISIC2018_Training_GroundTruth.csv')
labels = np.array(df.iloc[:, 1:]).astype(int)

# image_names = []
im_idx = []
for i in range(labels.shape[1]):
    print('Class {} instances: {}'.format(i+1, np.count_nonzero(np.where(labels[:, i] == 1))))
    indexes = np.argwhere(labels[:, i] == 1)
    indexes = indexes.squeeze()
    im_idx.append(indexes)
    # image_names.append(df['image'][indexes].tolist())



#%% Augmentation (DONE)
#augment_data(5, im_idx, df)
#augment_data(6, im_idx, df)
#augment_data(3, im_idx, df)
#augment_data(2, im_idx, df)
#augment_data(0, im_idx, df)
#augment_data(4, im_idx, df)
