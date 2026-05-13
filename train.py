# -*- coding: utf-8 -*-
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.train(data="datasets/FireData/data.yaml", epochs=250, batch=4)


if __name__ == "__main__":
    main()
