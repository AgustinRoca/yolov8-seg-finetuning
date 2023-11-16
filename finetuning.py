from ultralytics import YOLO

def main():
    model = YOLO('yolov8n-seg.pt', 'segment')  # load a pretrained model (recommended for training)

    # Train the model
    # results = model.train(data='coco-seg.yaml', epochs=100, imgsz=640)
    train_percentage = 0.8
    test_percentage = 0.1
    val_percentage = 1 - train_percentage - test_percentage
    dataset_name = f'venados_{train_percentage:.1f}-{val_percentage:.1f}'
    results = model.train(data=f'{dataset_name}-seg.yaml', epochs=100, imgsz=1280, 
                          flipud=0.5, 
                          degrees=45
                          )

if __name__ == '__main__':
    main()