from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO('yolov8n-seg.pt', 'segment')  # load a pretrained model (recommended for training)

    # Train the model
    # results = model.train(data='coco-seg.yaml', epochs=100, imgsz=640)
    results = model.train(data='venados-seg.yaml', epochs=100, imgsz=1280)

if __name__ == '__main__':
    main()