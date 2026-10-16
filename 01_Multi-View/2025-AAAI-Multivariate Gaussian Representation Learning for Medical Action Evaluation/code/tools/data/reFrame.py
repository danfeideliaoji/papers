import cv2
import os

def convert_video_fps(input_video_path, output_video_path, target_fps):
    subfolders = os.listdir(input_video_path)
    for subfolder in subfolders:
        if "front" in subfolder:
            continue
        subsubfolder = os.path.join(input_video_path, subfolder)
        subsubfolders = os.listdir(subsubfolder)
        for subsubsubfolder in subsubfolders:
            videos = os.listdir(os.path.join(subsubfolder, subsubsubfolder))
            for video in videos:
                video_path = os.path.join(subsubfolder, subsubsubfolder, video)
                print(video_path)
                cap = cv2.VideoCapture(video_path)
                
                original_fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                print(f"Original FPS: {original_fps} FPS")
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path.replace('RawData_revised', 'RawData_25FPS'), fourcc, target_fps, (width, height))

                frame_skip = int(original_fps / target_fps)
                
                current_frame = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if current_frame % frame_skip == 0:
                        out.write(frame)
                    
                    current_frame += 1

                cap.release()
                out.release()
                print(f"Conversion completed, saved as: {video_path.replace('RawData_revised', 'RawData_25FPS')}")

if __name__ == '__main__':
    input_video_path = '../RawData_revised'
    output_video_path = '../RawData_25FPS'
    target_fps = 25

    convert_video_fps(input_video_path, output_video_path, target_fps)