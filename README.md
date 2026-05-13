# Fire and Smoke Detection

This project uses YOLOv8 to detect fire and smoke in images or videos.

## Project Structure

```text
.
├── fire_detector.py          # Inference script for images and videos
├── train.py                  # YOLOv8 training entry point
├── requirements.txt          # Python dependencies
├── models/best.pt            # Trained model weights
├── Font/platech.ttf          # Font used for labels
└── datasets/FireData/data.yaml
```

Training images, labels, cache files, and training outputs are intentionally ignored by Git because they can be large. Put your dataset under `datasets/FireData/` when training locally.

## Installation

```bash
pip install -r requirements.txt
```

## Detect an Image

```bash
python fire_detector.py path/to/image.jpg output.jpg
```

## Detect a Video

```bash
python fire_detector.py path/to/video.mp4 output.mp4
```

## Use a Custom Model

```bash
python fire_detector.py path/to/image.jpg output.jpg --model models/best.pt --conf 0.25 --iou 0.7
```

## Train

Place the dataset at:

```text
datasets/FireData/
├── train/images
├── train/labels
├── valid/images
└── valid/labels
```

Then run:

```bash
python train.py
```

## Notes for GitHub

- `runs/`, Python cache files, and the full dataset are excluded from Git.
- `yolov8n.pt` is excluded because it is a downloadable base model.
- `models/best.pt` is kept so the detector can run after cloning, as long as the file stays below GitHub's file size limit.
