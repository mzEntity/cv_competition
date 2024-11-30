import os
import cv2
import torch
from pathlib import Path
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from models.experimental import attempt_load
from utils.torch_utils import select_device


def detect_and_crop(source_folder, save_folder, weights='runs/train/exp16/weights/best.pt', img_size=640,
                    conf_thres=0.25, iou_thres=0.45):
    # Directories
    save_dir = Path(save_folder)
    save_dir.mkdir(parents=True, exist_ok=True)  # Make save directory

    # Initialize device
    device = select_device('')  # Automatically select device, prefer GPU, otherwise use CPU
    model = attempt_load(weights, map_location=device)  # Load model
    model.eval()  # Set model to evaluation mode
    half = device.type != 'cpu'  # Half precision for CUDA
    if half:
        model.half()  # Convert model to FP16

    # Get class names from the model
    names = model.module.names if hasattr(model, 'module') else model.names  # Get class names

    # Load image files from the source folder
    image_files = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(source_folder, image_file)
        img = cv2.imread(image_path)  # Read image
        im0s = img.copy()  # Original image for display and saving results

        # Prepare the image for inference
        img_resized = cv2.resize(img, (img_size, img_size))  # Resize to model input size
        img_tensor = torch.from_numpy(img_resized).to(device).float()  # Convert to tensor
        img_tensor /= 255.0  # Normalize to [0, 1]
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # Convert to NCHW format

        # Run inference
        with torch.no_grad():
            pred = model(img_tensor)[0]  # Get predictions from the model

        # Apply NMS (Non-Maximum Suppression) to filter results
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for det in pred:  # Process each detection (there can be multiple detections)
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0s.shape).round()  # Rescale boxes

                for *xyxy, conf, cls in reversed(det):
                    # Convert xyxy (bounding box coordinates) to xywh format
                    xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()

                    # Crop the detected object (no resizing here, we keep the original size)
                    x1, y1, x2, y2 = map(int, xyxy)  # Convert to integer coordinates

                    # Check if the coordinates are valid (within image boundaries)
                    x1, y1 = max(0, x1), max(0, y1)  # Ensure coordinates are within image bounds
                    x2, y2 = min(im0s.shape[1], x2), min(im0s.shape[0], y2)

                    # Only crop if the coordinates are valid
                    if x2 > x1 and y2 > y1:
                        cropped_obj = im0s[y1:y2, x1:x2]  # Crop the region containing the object

                        # Save cropped object
                        cropped_image_name = f"{Path(image_file).stem}_{int(cls)}_{names[int(cls)]}_{conf:.2f}.jpg"
                        cropped_image_path = save_dir / cropped_image_name
                        cv2.imwrite(str(cropped_image_path), cropped_obj)  # Save the cropped object as an image
                    else:
                        print(f"Invalid cropping coordinates for image {image_file}, skipping this detection.")

        print(f"Processed image: {image_file} - Cropped objects saved to {save_dir}")

    print(f"Processed {len(image_files)} images. Results saved to {save_dir}")


def main():
    # Set parameters
    source_folder = 'trainset'  # Change this to your source folder
    save_folder = 'Img_save'  # Change this to your save folder
    weights = 'runs/train/exp16/weights/best.pt'  # Path to the model weights
    img_size = 640  # Image size for inference
    conf_thres = 0.25  # Confidence threshold for object detection
    iou_thres = 0.45  # IoU threshold for NMS

    # Run the detection and cropping function
    detect_and_crop(source_folder, save_folder, weights, img_size, conf_thres, iou_thres)


if __name__ == '__main__':
    main()
