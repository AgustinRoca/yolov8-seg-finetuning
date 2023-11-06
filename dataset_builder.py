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

    paths = create_directories(dataset_name)

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

    create_txts(train_data, 'train', imgs_path, mask_path, paths)
    create_txts(val_data, 'val', imgs_path, mask_path, paths)
    create_txts(test_data, 'test', imgs_path, mask_path, paths)
    

def create_txts(train_data, category, imgs_path, mask_path, paths):
    for img, mask in train_data:
        shutil.copy(os.path.join(imgs_path, img), os.path.join(paths['images'][category], img))
        create_label(os.path.join(mask_path, mask), os.path.join(paths['labels'][category], mask))

    train_txt_path = os.path.join(paths['dataset'], f'{category}.txt')
    with open(train_txt_path, 'w') as f:
        for img in os.listdir(paths['images'][category]):
            f.write(os.path.join(paths['images'][category], img) + '\n')

def create_directories(dataset_name):
    paths = {}
    paths['dataset'] = os.path.join('datasets', dataset_name)
    os.makedirs(paths['dataset'], exist_ok=True)

    paths['images'] = create_train_val_test_dirs(paths['dataset'], 'images')
    paths['labels'] = create_train_val_test_dirs(paths['dataset'], 'labels')

    return paths

def create_train_val_test_dirs(dataset_path,root_path):
    dirs = {}
    dataset_root_path = os.path.join(dataset_path, root_path)
    os.makedirs(dataset_root_path, exist_ok=True)
    train_path = os.path.join(dataset_root_path, 'train')
    val_path = os.path.join(dataset_root_path, 'val')
    test_path = os.path.join(dataset_root_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    dirs['train'] = train_path
    dirs['val'] = val_path
    dirs['test'] = test_path
    return dirs

def create_label(image_path, label_path):
    """
    It creates a label file for a given image.

    Parameters
    ----------
    image_path : str
        Image with the segmentation.
    label_path : str
        Output file.
    """
    arr = np.asarray(Image.open(image_path))

    # There may be a better way to do it, but this is what I have found so far
    cords = list(features.shapes(arr, mask=(arr >0)))[0][0]['coordinates'][0]
    label_line = '0 ' + ' '.join([f'{int(cord[0])/arr.shape[1]} {int(cord[1])/arr.shape[0]}' for cord in cords]) # TODO: Check if the order is correct

    with open(label_path.split('_')[0] + '.txt', "w+") as f:
        f.write(label_line + '\n')

if __name__ == '__main__':
    build_dataset('venados', 'imgs', 'ground_truth', 0.12, 0)