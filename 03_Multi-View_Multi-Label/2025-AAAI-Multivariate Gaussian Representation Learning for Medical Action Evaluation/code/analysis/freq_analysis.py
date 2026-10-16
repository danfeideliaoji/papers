import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt, stft
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
import pywt

def read_json_files(file_name):
    with open(file_name,'r', encoding='UTF-8') as f:
        load_dict = json.load(f)
    return load_dict

def extract_keypoints(data):
    time_series = {}
    for frame in data:
        if frame['frame_id'] >= 60 and frame['frame_id'] <= 300:
            frame_id = frame['frame_id']
            for instance_id, instance in enumerate(frame['instances']):
                if instance_id not in time_series:
                    time_series[instance_id] = {'xy': [], 'scores': []}
                keypoints = np.array(instance['keypoints'])
                combined_points = keypoints[:, :2]
                time_series[instance_id]['xy'].append([combined_points[91],combined_points[112]])
                time_series[instance_id]['scores'].append(instance['keypoint_scores'])
    return time_series

def smooth_data(data, window_size=2):
    smoothed_data = np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='valid'), axis=0, arr=data)
    return smoothed_data

def exponential_smoothing(data, alpha):
    smoothed_data = [data[0]]  
    for i in range(1, len(data)):
        smoothed_data.append(alpha * data[i] + (1 - alpha) * smoothed_data[-1])
    smoothed_data = np.array(smoothed_data)
    return smoothed_data

def median_filter(data, window_size=5):
    return medfilt(data, kernel_size=window_size)

def gaussian_smoothing(data, sigma):
    return gaussian_filter1d(data, sigma)

def apply_pca(data):
    pca = PCA(n_components=1)
    return pca.fit_transform(data)

def find_dominant_frequencies(freqs, Sxx, threshold=0.6):
    dominant_frequencies = []
    for i in range(Sxx.shape[1]):
        mask = freqs > threshold
        filtered_freqs = freqs[mask]
        filtered_magnitudes = Sxx[mask, i]
        if len(filtered_magnitudes) > 0:
            peak_index = np.argmax(filtered_magnitudes)
            dominant_frequencies.append(filtered_freqs[peak_index])
        else:
            dominant_frequencies.append(0)
    return dominant_frequencies

def apply_cwt(data):
    wavename = 'cmor1.5-1.5'
    totalscal = 256
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / fps)
    mask = frequencies > 0.8
    cwtmatr_masked = cwtmatr[mask, :]
    maxfreq = []
    for i in range(cwtmatr.shape[1]):
        if frequencies[np.argmax(abs(cwtmatr_masked[:, i]))] > 0.8:
            maxfreq.append(frequencies[np.argmax(abs(cwtmatr_masked[:, i]))])
        else:
            maxfreq.append(0)
    return cwtmatr, frequencies, maxfreq

def visualize_time_series(time_series, fps, instance_id=0):
    plt.figure(figsize=(14, 7))
    xy_data = np.array(time_series[instance_id]['xy'])

    xy_smoothed = smooth_data(xy_data, window_size=2)
    xy_pca = apply_pca(xy_smoothed.reshape(xy_smoothed.shape[0], -1))

    plt.subplot(3, 1, 1)
    plt.plot(xy_pca, label='Combined XY PCA')
    plt.xlabel('Frame')
    plt.ylabel('PCA Value')
    plt.legend()
    plt.title(f'Person {instance_id} Time Series')

    cwtmatr, f, maxfreq = apply_cwt(xy_pca.flatten())

    t = np.arange(0, xy_pca.shape[0] / fps, 1 / fps)
    plt.subplot(3, 1, 2)
    plt.contourf(t, f, abs(cwtmatr))
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Person {instance_id} CWT')

    plt.subplot(3, 1, 3)
    plt.plot(t, maxfreq, label='Dominant Frequency')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.title(f'Person {instance_id} Dominant Frequency Over Time')

    plt.tight_layout()
    plt.show()

file_name = 'test3type.json' 
data = read_json_files(file_name)
time_series = extract_keypoints(data)
fps = 30
visualize_time_series(time_series, fps, instance_id=0)