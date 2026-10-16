import csv
import json
import os
import cv2
import random
from collections import defaultdict
import mmcv
import numpy as np
import pickle

label_mapping = {
    "Correct": 0,
    "Depth-Insufficient Press": 1,
    "Depth-Excess": 2,
    "Depth-Insufficient Rebound": 3,
    "Freq-High Frequency": 4,
    "Freq-Low Frequency": 5,
    "Freq-Excessive Momentum": 6,
    "Position-Wrong Position": 7,
    "Position-Movement": 8,
    "Hand-Double Hand-Cross": 9,
    "Hand-Double Hand-Overlap": 10,
    "Hand-Double Hand-Two Fist": 11,
    "Hand-One Hand-Fist": 12,
    "Hand-One Hand-Hand": 13,
    "Angle-Arm Bend": 14,
    "Angle-Left Skew": 15,
    "Angle-Right Skew": 16,
    "Angle-Front Skew": 17,
    "Angle-Back Skew": 18,
    "Body-Waist Unstable": 19,
    "Body-Thigh Antevert": 20,
    "Body-Thigh Tilt Back": 21
}

csv_file_path = '../CPREval_Annotation.csv'
base_video_path = '../RawData_25FPS/'
label_video_dict = defaultdict(list)
video_list = []

with open(csv_file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)

    for row in csvreader:
        label = row[1]
        front_video_path = row[8]
        side_video_path = row[9]

        label_index = label_mapping.get(label, -1)

        if label_index != -1:
            for video_path in [front_video_path, side_video_path]:
                full_video_path = os.path.join(base_video_path, video_path)

                vid_name = os.path.splitext(video_path)[0]

                cap = cv2.VideoCapture(full_video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                video_entry = {
                    "vid_name": vid_name,
                    "label": label_index,
                    "start_frame": 0,
                    "end_frame": frame_count - 1
                }

                video_list.append(video_entry)

                label_video_dict[label_index].append(video_entry)

def split_train_test(label_video_dict, train_ratio=0.85):
    train_videos = []
    test_videos = []

    for label, videos in label_video_dict.items():
        random.shuffle(videos)
        split_index = int(train_ratio * len(videos))

        train_videos.extend(videos[:split_index])
        test_videos.extend(videos[split_index:])
    
    return train_videos, test_videos

for label, video in label_video_dict.items():
    key = next((k for k, v in label_mapping.items() if v == label), None)
    print(key,len(video))

train_videos, test_videos = split_train_test(label_video_dict, train_ratio=0.8)

complete_json_path = './video_list.json'
train_json_path = './train_list.json'
test_json_path = './test_list.json'

with open(complete_json_path, 'w') as jsonfile:
    json.dump(video_list, jsonfile, indent=4)

with open(train_json_path, 'w') as jsonfile:
    json.dump(train_videos, jsonfile, indent=4)

with open(test_json_path, 'w') as jsonfile:
    json.dump(test_videos, jsonfile, indent=4)

print(f"Complete JSON file generated: {complete_json_path}")
print(f"Train JSON file generated: {train_json_path}")
print(f"Test JSON file generated: {test_json_path}")


def load_json(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e} in file {file_path}")
            return None
    
def get_video_shape(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_file}")
        return (480, 640)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return (height, width)

def create_annotations(pred_dir, video_list, video_base_dir, add_noise=False, noise_alpha=0.1):
    annotations = []
    
    for video_info in video_list:
        video_name = video_info['vid_name']
        label = video_info['label']
        start_frame = video_info['start_frame']
        end_frame = video_info['end_frame']

        json_file = os.path.join(pred_dir, f"{video_name}.json")
        if not os.path.exists(json_file):
            print(f"File not found: {json_file}")
            continue

        data = load_json(json_file)

        video_file = os.path.join(video_base_dir, f"{video_name}.MP4")
        
        img_shape = get_video_shape(video_file)

        num_person_raw = max(len(frame['instances']) for frame in data)
        total_frames = end_frame - start_frame + 1

        keypoints = np.zeros((num_person_raw, total_frames, 133, 2), dtype=np.float16)
        kp_scores = np.zeros((num_person_raw, total_frames, 133), dtype=np.float16)

        for frame_id, frame in enumerate(data[start_frame:end_frame+1]):
            instances = frame['instances']
            for person_idx, instance in enumerate(instances):
                kp = instance['keypoints']
                kp_score = instance['keypoint_scores']
                if len(kp) == 133 and len(kp_score) == 133:
                    keypoints[person_idx, frame_id] = np.array(kp)
                    kp_scores[person_idx, frame_id] = np.array(kp_score)
        
        if add_noise:
            keypoints_noise = np.random.normal(0, noise_alpha, keypoints.shape)
            keypoints = keypoints + keypoints_noise
            
            scores_noise = np.random.normal(0, noise_alpha, kp_scores.shape)
            kp_scores = np.clip(kp_scores + scores_noise, 0.0, 1.0)

        annotation = {
            'frame_dir': video_name,
            'label': label,
            'img_shape': img_shape,  
            'original_shape': img_shape,  
            'total_frames': total_frames,
            'num_person_raw': num_person_raw,
            'keypoint': keypoints,
            'keypoint_score': kp_scores
        }
        annotations.append(annotation)

    return annotations

def create_pkl(train_json_file, test_json_file, pred_dir, video_base_dir, output_pkl):
    train_list = load_json(train_json_file)
    test_list = load_json(test_json_file)

    split = {
        'train': [video['vid_name'] for video in train_list],
        'test': [video['vid_name'] for video in test_list]
    }

    annotations = []
    annotations.extend([])
    annotations.extend(create_annotations(pred_dir, test_list, video_base_dir, noise_alpha=0.05))

    pkl_data = {
        'split': split,
        'annotations': annotations
    }

    with open(output_pkl, 'wb') as f:
        pickle.dump(pkl_data, f)

    print(f"PKL file saved: {output_pkl}")

train_json_file = './train_list.json'
test_json_file = './test_list.json'
pred_dir = '../pred'
video_base_dir = '../RawData_25FPS'
output_pkl = './custom_dataset_w_kpscores_alpha_005.pkl'

create_pkl(train_json_file, test_json_file, pred_dir, video_base_dir, output_pkl)