from ultralytics import YOLO
import os
from tqdm import tqdm

def main():
    model = YOLO('runs/segment/train/weights/best.pt')

    test_imgs = os.listdir('datasets/venados_0.8-0.1/images/test')
    deers_found = 0
    for img in tqdm(test_imgs):
        result = model.predict(os.path.join('imgs', img), save=True)
        if result[0].masks:
            deers_found += 1
    print(f'Imagenes con venados encontrados: {deers_found} / {len(test_imgs)}')
    print(f'Porcentaje del total: {deers_found / len(test_imgs) * 100:.2f}%')

if __name__ == '__main__':
    main()