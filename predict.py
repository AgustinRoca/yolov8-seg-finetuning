from ultralytics import YOLO
import os
from tqdm import tqdm

def main():
    model = YOLO('runs/segment/train/weights/best.pt')

    train_imgs = os.listdir('ground_truth')
    train_imgs = [img.split('_')[0] for img in train_imgs]
    imgs = os.listdir('imgs')
    imgs = [img for img in imgs if img.split('.')[0] not in train_imgs]
    deers_found = 0
    for img in tqdm(imgs):
        result = model.predict(os.path.join('imgs', img), save=True)
        if result[0].masks:
            deers_found += 1
    print(f'Imagenes con venados encontrados: {deers_found} / {len(imgs)}')
    print(f'Porcentaje del total: {deers_found / len(imgs) * 100:.2f}%')

if __name__ == '__main__':
    main()