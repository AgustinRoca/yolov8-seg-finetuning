import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

def main():
    imgs = os.listdir('runs/segment/predict')
    for img_name in tqdm(imgs):
        img_path = os.path.join('imgs', img_name)
        prediction_path = os.path.join('runs/segment/predict', img_name)
        img = Image.open(img_path)
        prediction = Image.open(prediction_path)
        fig = plt.figure(figsize=(20, 10))
        st = fig.suptitle(img_name, fontsize="x-large")
        ax1 = plt.subplot(121)
        plt.imshow(img)
        ax2 = plt.subplot(1,2,2, sharex=ax1, sharey=ax1)
        plt.imshow(prediction)
        plt.show()

if __name__ == '__main__':
    main()