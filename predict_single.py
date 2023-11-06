from ultralytics import YOLO
import os
from tqdm import tqdm

def main():
    # Load a model
    model = YOLO('runs/segment/train/weights/best.pt')

    # Run batched inference on a list of images

    train_imgs = os.listdir('ground_truth')
    train_imgs = [img.split('_')[0] for img in train_imgs]
    imgs = os.listdir('imgs')
    imgs = [img for img in imgs if img.split('.')[0] not in train_imgs]
    for img in tqdm(imgs):
        model.predict(os.path.join('imgs', img), save=True)

if __name__ == '__main__':
    main()