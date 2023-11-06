import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

def main():
    imgs = os.listdir('runs/segment/predict')
    for img in tqdm(imgs):
        img_path = os.path.join('imgs', img)
        prediction_path = os.path.join('runs/segment/predict', img)
        img = Image.open(img_path)
        prediction = Image.open(prediction_path)
        plt.figure(figsize=(20, 10))
        ax1 = plt.subplot(121)
        plt.imshow(img)
        ax2 = plt.subplot(1,2,2, sharex=ax1, sharey=ax1)
        plt.imshow(prediction)
        plt.show()

if __name__ == '__main__':
    main()