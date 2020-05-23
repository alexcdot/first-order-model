import os

worker_id = int(os.environ['WORKER_ID'])
assert(os.environ.get('CUDA_VISIBLE_DEVICES') is not None)
os.environ['RUN_FER'] = '0'

from server import run_cloning, adjust_clip

fps = 4
if os.environ.get('FPS') is not None:
    fps = int(os.environ['FPS'])
# max time
t = 60
target_resolution = (256, 256)

classes = ['frosh', 'smore', 'junior', 'senior']
my_class = classes[worker_id % 4]

data_dir = 'data'

imgs_dir = os.path.join(data_dir, 'blacker_pics', my_class + '_pics_files')
all_img_names = os.listdir(imgs_dir)
my_img_names = [img for img in all_img_names if 'jpg' in img
                and int(img.split('_')[1].split('.')[0]) % 2 == worker_id // 4]

print("my imgs:", my_img_names)

time_id = str((worker_id % 4) * 45) + '_' + str((worker_id % 4) * 45 + 45)
driving_video_path = os.path.join(data_dir, 'still_alive_' + time_id + '.mp4')

result_video_dir = os.path.join(data_dir, my_class + '_videos')

if not os.path.exists(result_video_dir):
    os.mkdir(result_video_dir)

resized_clip = adjust_clip(driving_video_path, fps, t, target_resolution)

for img in my_img_names:
    img_id = time_id + '_' + str(fps) + '_' + str(int(img.split('_')[1].split('.')[0]))
    source_image_path = os.path.join(imgs_dir, img)
    result_video_path = os.path.join(result_video_dir, 'still_alive_' + img_id + '.mp4')
    
    if not os.path.exists(result_video_path):
        run_cloning(source_image_path, driving_video_path,
            result_video_path, run_fer=False, fps=fps, t=t, resized_clip=resized_clip)
    