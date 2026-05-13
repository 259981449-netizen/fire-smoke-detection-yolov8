# -*- coding: utf-8 -*-
"""
Fire and smoke detection with a YOLOv8 model.

Usage:
    python fire_detector.py <input_path> [output_path]

The input can be an image (.jpg, .png, .jpeg, .bmp, .tiff) or a video
(.mp4, .avi, .mov, .mkv, .wmv). If output_path is provided, the annotated
image or video will be saved there.
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


class FireDetector:
    def __init__(self, model_path="models/best.pt", conf=0.25, iou=0.7):
        self.conf = conf
        self.iou = iou

        self.class_names = {0: "Fire", 1: "Smoke"}
        self.display_names = ["Fire", "Smoke"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path, task="detect")

        self.model(np.zeros((48, 48, 3)), device=self.device)
        print("Model loaded")

        try:
            self.font = ImageFont.truetype("Font/platech.ttf", 25, 0)
        except OSError:
            self.font = ImageFont.load_default()

        self.colors = [
            (255, 0, 0),      # Fire
            (128, 128, 128),  # Smoke
        ]

    def detect_image(self, image_path):
        print(f"Detecting image: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")

        start_time = time.time()
        results = self.model(img, conf=self.conf, iou=self.iou)[0]
        detect_time = time.time() - start_time

        detections = self._parse_results(results)
        result_img = self._draw_detections(img.copy(), detections)

        return {
            "type": "image",
            "path": image_path,
            "detections": detections,
            "detect_time": detect_time,
            "original_image": img,
            "result_image": result_img,
        }

    def detect_video(self, video_path):
        print(f"Detecting video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video info: {width}x{height}, {fps:.1f} fps, {total_frames} frames")

        all_detections = []
        frame_count = 0
        total_detect_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}", end="\r")

            start_time = time.time()
            results = self.model(frame, conf=self.conf, iou=self.iou)[0]
            detect_time = time.time() - start_time
            total_detect_time += detect_time

            detections = self._parse_results(results)
            result_frame = self._draw_detections(frame.copy(), detections)

            all_detections.append(
                {
                    "frame_id": frame_count,
                    "detections": detections,
                    "detect_time": detect_time,
                    "original_frame": frame,
                    "result_frame": result_frame,
                }
            )

        cap.release()

        print(f"\nVideo detection completed, processed {frame_count} frames")
        return {
            "type": "video",
            "path": video_path,
            "total_frames": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "all_detections": all_detections,
            "avg_detect_time": total_detect_time / frame_count if frame_count > 0 else 0,
            "total_detect_time": total_detect_time,
        }

    def _parse_results(self, results):
        detections = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.tolist()
            classes = results.boxes.cls.tolist()
            confs = results.boxes.conf.tolist()

            for box, cls, conf in zip(boxes, classes, confs):
                class_id = int(cls)
                detections.append(
                    {
                        "bbox": [int(coord) for coord in box],
                        "class_id": class_id,
                        "class_name": self.class_names.get(class_id, str(class_id)),
                        "display_name": self.display_names[class_id],
                        "confidence": float(conf),
                    }
                )

        return detections

    def _draw_detections(self, img, detections):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        for detection in detections:
            bbox = detection["bbox"]
            class_id = detection["class_id"]
            conf = detection["confidence"]
            label = f"{detection['display_name']} {conf:.2f}"
            color = self.colors[class_id % len(self.colors)]

            draw.rectangle(bbox, outline=color, width=2)

            label_y = max(0, bbox[1] - 25)
            text_bbox = draw.textbbox((bbox[0], label_y), label, font=self.font)
            draw.rectangle([bbox[0] - 1, label_y, text_bbox[2] + 2, text_bbox[3]], fill=color)
            draw.text((bbox[0] + 2, label_y + 2), label, fill=(255, 255, 255), font=self.font)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def print_results(self, result):
        if result["type"] == "image":
            print("\n=== Image Detection Result ===")
            print(f"File: {result['path']}")
            print(f"Detection time: {result['detect_time']:.3f} s")
            print(f"Objects detected: {len(result['detections'])}")

            if result["detections"]:
                print("\nDetails:")
                for i, det in enumerate(result["detections"], 1):
                    print(
                        f"{i}. {det['display_name']} - confidence: "
                        f"{det['confidence']:.2f} - bbox: {det['bbox']}"
                    )

        elif result["type"] == "video":
            print("\n=== Video Detection Result ===")
            print(f"File: {result['path']}")
            print(f"Total frames: {result['total_frames']}")
            print(f"FPS: {result['fps']:.1f}")
            print(f"Average detection time: {result['avg_detect_time']:.3f} s/frame")
            print(f"Total detection time: {result['total_detect_time']:.3f} s")

            class_counts = {}
            for frame_result in result["all_detections"]:
                for det in frame_result["detections"]:
                    cls_name = det["display_name"]
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            print("\nClass counts:")
            for cls_name, count in class_counts.items():
                print(f"{cls_name}: {count}")

    def save_result(self, result, output_path):
        if result["type"] == "image":
            cv2.imwrite(output_path, result["result_image"])
            print(f"Result saved to: {output_path}")

        elif result["type"] == "video":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, result["fps"], (result["width"], result["height"]))

            for frame_result in result["all_detections"]:
                out.write(frame_result["result_frame"])

            out.release()
            print(f"Result video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fire and smoke detection")
    parser.add_argument("input_path", help="Input image or video path")
    parser.add_argument("output_path", nargs="?", help="Optional output file path")
    parser.add_argument("--model", default="models/best.pt", help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: input file does not exist: {args.input_path}")
        return

    detector = FireDetector(model_path=args.model, conf=args.conf, iou=args.iou)

    try:
        input_path_lower = args.input_path.lower()
        if input_path_lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            result = detector.detect_image(args.input_path)
        elif input_path_lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv")):
            result = detector.detect_video(args.input_path)
        else:
            print("Error: unsupported file format")
            return

        detector.print_results(result)

        if args.output_path:
            detector.save_result(result, args.output_path)

    except Exception as exc:
        print(f"Detection failed: {exc}")
        raise


if __name__ == "__main__":
    main()
