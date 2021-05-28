import os
import cv2
import argparse
import subprocess
import pathlib
import shutil

from predict import PoseBaselineForCOCO
from src.cmu2kinect import CMUPose2KinectData

def run_openpose(openpose_path, video_path, save_json_path):
    p = pathlib.Path(save_json_path)
    if not p.is_absolute():
        save_json_path = p.resolve()
    save_json_path = str(save_json_path)
    if " " in str(save_json_path):
        save_json_path = "\"" + str(save_json_path) + "\""    
    
    p = pathlib.Path(video_path)
    if not p.is_absolute():
        video_path = p.resolve()
    video_path = str(video_path)
    if " " in str(video_path):
        video_path = "\"" + str(video_path) + "\""

    if not os.path.exists(save_json_path):
        os.makedirs(save_json_path)

    cd = os.getcwd()
    os.chdir(openpose_path)

    # call bin\OpenPoseDemo.exe --video %video% --write_images %imgs_dir% --write_json %pose_dir% --no_gui_verbose -display 0 -render_pose 2 --number_people_max 1 
    exe = []
    exe.append('bin\\OpenPoseDemo.exe')
    exe.append('--video')
    exe.append(video_path)
    exe.append('--write_json')
    exe.append(save_json_path)
    # exe.append('--write_images')
    # exe.append(save_image_path)
    exe.append('--render_pose')
    exe.append('0')
    exe.append('--no_gui_verbose')
    exe.append('-display')
    exe.append('0')
    exe.append('--number_people_max')
    exe.append('1')
    exe.append('--num_gpu')
    exe.append('1')
    exe.append('--num_gpu_start')
    exe.append('0')

    print(' '.join(exe))
    subprocess.call(' '.join(exe), shell=True)

    os.chdir(cd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run openpose')

    parser.add_argument('video_path', default='D:/TED_videos/VideoStorage/clip_videos/_2PhFnwj608/_2PhFnwj608_0.mp4', help='')
    parser.add_argument('csv_path', default='./tmp/sample.csv', help='')
    parser.add_argument('--openpose_path', default='./openpose-1.4.0-win64-gpu-binaries/', help='')
    parser.add_argument('--checkpoint_path', default='./checkpoint/15joints_best.chkpt', help='')

    args = parser.parse_args()

    video_path = args.video_path
    csv_path = args.csv_path
    openpose_path = args.openpose_path
    checkpoint_path = args.checkpoint_path

    # OpenPose
    print('Start OpenPose')
    openpose_output_dir = "./tmp/json/"
    if os.path.exists(openpose_output_dir):
        shutil.rmtree(openpose_output_dir)
    run_openpose(openpose_path, video_path, openpose_output_dir)

    # 3D-Pose-Baseline
    print('Start 3D-Pose-Baseline')
    p = PoseBaselineForCOCO(checkpoint_path)
    pose2d, pose3d = p.predict(openpose_output_dir, mode='joint15')

    # get fps
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    CMUPose2KinectData(pose3d, save_csv=csv_path, fps=fps)
    print('Finish.')

    shutil.rmtree(openpose_output_dir)

    print('Output to {}'.format(csv_path))







