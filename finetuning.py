from ultralytics import YOLO

def main():
    model = YOLO('yolov8n-seg.pt', 'segment')  # load a pretrained model (recommended for training)

    # Train the model
    # results = model.train(data='coco-seg.yaml', epochs=100, imgsz=640)
    val_percentage = 0.9
    dataset_name = f'venados_{1-val_percentage:.1f}-{val_percentage:.1f}'
    results = model.train(data=f'{dataset_name}-seg.yaml', epochs=100, imgsz=1280, 
                          flipud=0.5, 
                          degrees=45,
                          )

if __name__ == '__main__':
    main()