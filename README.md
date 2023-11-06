# Finetune YOLOv8-seg with masks

## Setup

1. Clone the repository

    ```bash
    git clone https://github.com/AgustinRoca/yolov8-seg-finetuning.git
    ```

2. Install dependencies

    ```bash
    pip3 install -r requirements.txt
    ```

## Prepare dataset

1. Put the input images in `imgs/` and the corresponding masks in `ground_truth/`.

2. Create a file `dataset.yaml` in the root directory of the repository with the following content:
    ```yaml
    path: ../datasets/venados  # dataset root dir
    train: images/train  # train images (relative to 'path') 4 images
    val: images/val  # val images (relative to 'path') 4 images

    # Classes
    names:
    0: venado
    ```
    In the classes section, you have to enumerate all the classes that you want to detect. In this case, we only have one class, so we only have to add `0: venado`. The number is the class id, and the name is the class name. The class name is only used for visualization purposes.

3. Check `create_label` function from `dataset_builder.py` to see how the masks are generated. The current implementation assumes that the masks are binary images, where the background is black and the foreground is white. WHITE=DEER PRESENT, BLACK=DEER ABSENT. The format of the output `.txt` files should be:
    ```text
    <class id> <x1> <y1> <x2> <y2> <x3> <y3> ... <xn> <yn>
    ```
    where `n` is the number of points that define the polygon.

    The (x, y) coordinates are numbers between 0 and 1, relative to the image size.

4. Run `python3 dataset_builder.py` to generate the dataset


## Train

1. Check that the `data` argument in the train function of `finetuning.py` is correct. It should point to the `dataset.yaml` file that you created in the second step of the previous section.

2. Run `python3 finetuning.py` to train the model. The model will be saved in `runs/segment/train/weights/best.pt`

## Validate

1. Check that the model path points to the trained model.

2. Run `python3 validation.py` to validate the model. The results will be saved in `runs/segment/val/`

## Predict

1. Check that the model path points to the trained model.

2. Use the following function:

    ```python
    model = YOLO('runs/segment/train/weights/best.pt')
    model.predict(path_to_img) # Can add save=True to save the output image
    ```

## Visualize

1. Check the paths. One should be the prediction output images directory, and the other should be the input images directory.

2. Run `python3 view_side_by_side.py` to visualize the results.
