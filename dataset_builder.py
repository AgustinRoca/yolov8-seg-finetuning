import os
import shutil
from rasterio import features
from PIL import Image
import numpy as np

def build_dataset(dataset_name, imgs_path, mask_path, validation_percentage=0.5, test_percentage=0):
    """
    It transforms plain images and masks to a YOLOv8 dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to build.
    imgs_path : str
        Path to the images.
    mask_path : str
        Path to the masks.
    validation_percentage : float, optional
        Percentage of the dataset to use for validation. The default is 0.5.
    test_percentage : float, optional
        Percentage of the dataset to use for testing. The default is 0.
    """

    # Create the dataset directory
    dataset_path = os.path.join('datasets', dataset_name)
    os.makedirs(dataset_path, exist_ok=True)

    # Create the train and val directories
    dataset_imgs_path = os.path.join(dataset_path, 'images')
    dataset_labels_path = os.path.join(dataset_path, 'labels')
    os.makedirs(dataset_imgs_path, exist_ok=True)
    os.makedirs(dataset_labels_path, exist_ok=True)

    # Create the images and labels directories
    train_imgs_path = os.path.join(dataset_imgs_path, 'train')
    train_labels_path = os.path.join(dataset_labels_path, 'train')
    val_imgs_path = os.path.join(dataset_imgs_path, 'val')
    val_labels_path = os.path.join(dataset_labels_path, 'val')
    test_imgs_path = os.path.join(dataset_imgs_path, 'test')
    test_labels_path = os.path.join(dataset_labels_path, 'test')
    os.makedirs(train_imgs_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(val_imgs_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)
    os.makedirs(test_imgs_path, exist_ok=True)
    os.makedirs(test_labels_path, exist_ok=True)

    all_masks = os.listdir(mask_path)
    all_imgs = [img.split('_')[0] + '.JPG' for img in all_masks]
    all_data = list(zip(all_imgs, all_masks))
    np.random.shuffle(all_data)
    validation_start_index = 0
    validation_end_index = int(len(all_data) * validation_percentage)
    test_start_index = validation_end_index
    test_end_index = test_start_index + int(len(all_data) * test_percentage)
    train_start_index = test_end_index
    train_end_index = len(all_data)

    val_data = all_data[validation_start_index:validation_end_index]
    test_data = all_data[test_start_index:test_end_index]
    train_data = all_data[train_start_index:train_end_index]

    # Copy the images and masks to the train, val and test directories
    for img, mask in train_data:
        shutil.copy(os.path.join(imgs_path, img), os.path.join(train_imgs_path, img))
        create_label(os.path.join(mask_path, mask), os.path.join(train_labels_path, mask))

    for img, mask in val_data:
        shutil.copy(os.path.join(imgs_path, img), os.path.join(val_imgs_path, img))
        create_label(os.path.join(mask_path, mask), os.path.join(val_labels_path, mask))

    for img, mask in test_data:
        shutil.copy(os.path.join(imgs_path, img), os.path.join(test_imgs_path, img))
        create_label(os.path.join(mask_path, mask), os.path.join(test_labels_path, mask))

    # Create the train.txt file
    train_txt_path = os.path.join(dataset_path, 'train.txt')
    with open(train_txt_path, 'w') as f:
        for img in os.listdir(train_imgs_path):
            f.write(os.path.join(train_imgs_path, img) + '\n')

    # Create the val.txt file
    val_txt_path = os.path.join(dataset_path, 'val.txt')
    with open(val_txt_path, 'w') as f:
        for img in os.listdir(val_imgs_path):
            f.write(os.path.join(val_imgs_path, img) + '\n')

    # Create the test.txt file
    test_txt_path = os.path.join(dataset_path, 'test.txt')
    with open(test_txt_path, 'w') as f:
        for img in os.listdir(test_imgs_path):
            f.write(os.path.join(test_imgs_path, img) + '\n')

def create_label(image_path, label_path):
    arr = np.asarray(Image.open(image_path))

    # There may be a better way to do it, but this is what I have found so far
    cords = list(features.shapes(arr, mask=(arr >0)))[0][0]['coordinates'][0]
    label_line = '0 ' + ' '.join([f'{int(cord[0])/arr.shape[1]} {int(cord[1])/arr.shape[0]}' for cord in cords]) # TODO: Check if the order is correct

    with open(label_path.split('_')[0] + '.txt', "w+") as f:
        f.write(label_line + '\n')

build_dataset('venados', 'imgs', 'ground_truth', 0.12, 0)