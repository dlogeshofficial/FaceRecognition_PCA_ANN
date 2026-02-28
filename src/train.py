import os
import cv2
import numpy as np

def load_dataset(dataset_path, img_size=(100, 100), exclude_people=None):
    if exclude_people is None:
        exclude_people = []
    
    # The dataset is inside 'dataset/faces'
    faces_path = os.path.join(dataset_path, "faces")
    if not os.path.exists(faces_path):
        faces_path = dataset_path # fallback

    images = []
    labels = []
    label_map = {}
    label_id = 0

    if not os.path.exists(faces_path):
        print(f"Error: {faces_path} does not exist.")
        return np.array([]), np.array([]), {}

    for person in os.listdir(faces_path):
        if person in exclude_people:
            continue
            
        person_path = os.path.join(faces_path, person)

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

    if len(images) == 0:
        return np.array([]), np.array([]), label_map
        
    X = np.array(images).T
    y = np.array(labels)

    return X, y, label_map
