import os
import cv2
import numpy as np

def load_dataset(dataset_path, img_size=(100, 100)):
    images = []
    labels = []
    label_map = {}
    label_id = 0

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)

        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, 0)

            if img is None:
                continue

            img = cv2.resize(img, img_size)
            img = img.flatten()
            images.append(img)
            labels.append(label_id)

        label_id += 1

    X = np.array(images).T
    y = np.array(labels)

    return X, y, label_map