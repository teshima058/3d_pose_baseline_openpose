import os
import argparse
import subprocess
import pathlib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run openpose')

    parser.add_argument('openpose_path', default='D:/Program Files/openpose-1.4.0-win64-gpu-binaries/', help='')
    parser.add_argument('video_path', default='D:/TED_videos/VideoStorage/clip_videos/_2PhFnwj608/_2PhFnwj608_0.mp4', help='')
    parser.add_argument('save_json_path', default='./tmp/json/', help='')

    args = parser.parse_args()

    openpose_path = args.openpose_path
    video_path = args.video_path
    save_json_path = args.save_json_path

    p = pathlib.Path(save_json_path)
    if not p.is_absolute():
        save_json_path = p.resolve()

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
    exe.append(str(save_json_path))
    # exe.append('--write_images')
    # exe.append(save_image_path)
    exe.append('--render_pose')
    exe.append('0')
    exe.append('--no_gui_verbose')
    exe.append('-display')
    exe.append('0')
    exe.append('--number_people_max')
    exe.append('1')

    print(' '.join(exe))
    subprocess.call(' '.join(exe), shell=True)

    os.chdir(cd)


