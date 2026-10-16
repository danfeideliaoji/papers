import numpy as np
from sklearn.metrics import average_precision_score
import mmcv


def ltAnalysis(mAP_cls, dataset):
    if dataset == 'coco-lt':
        freq_file = '../data/coco/class_freq.pkl'
        dataset_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
            'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
            'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
            ]
    elif dataset == 'voc-lt':
        freq_file = '/data/zfr888/ljt/voc/class_freq.pkl'
        dataset_classes = ["waste_bag", "metal", "shoe", "plastic", "bottle", "carton", "lile", "galss"]
    head = 0
    medium = 0
    tail = 0
    head_AP, middle_AP, tail_AP = {},{},{}
    head_count, medium_count, tail_count = 0,0,0
    
    class_freq = np.array([1117,329,310,1648,905,645,934,317])###trainval
    hids = set(np.where(class_freq>=1000)[0])
    mids = set(np.where((class_freq<1000) * (class_freq >= 500))[0])
    tids = set(np.where(class_freq<500)[0])
    
    
    # class_freq = np.array([288,89,74,410,241,159,211,84])###test
    # hids = set(np.where(class_freq>=250)[0])
    # mids = set(np.where((class_freq<250) * (class_freq >= 100))[0])
    # tids = set(np.where(class_freq<100)[0])
    
    
    for id in hids:
        head_AP[dataset_classes[id]] = mAP_cls[id]
        head += mAP_cls[id]
    for id in mids:
        middle_AP[dataset_classes[id]] = mAP_cls[id]
        medium += mAP_cls[id]
    for id in tids:
        tail_AP[dataset_classes[id]] = mAP_cls[id]
        tail += mAP_cls[id]
    head = head/len(hids)
    medium = medium/len(mids)
    tail = tail/len(tids)
    print(f'head APs: {head_AP}, middle APs: {middle_AP}, tail APs: {tail_AP}')
    print(f'mAP head: {head}, mAP medium: {medium}, mAP tail: {tail}')

def eval_map(results, gt_labels):
    "Evaluate mAP of a dataset"
    results = np.asarray(results)
    gt_labels = np.asarray(gt_labels)
    assert results.shape[0] == gt_labels.shape[0]
    eval_results = []
    num_samples, num_classes = results.shape
    APs = average_precision_score(gt_labels, results, average=None)
    mAP = APs.mean()
    return mAP, APs
