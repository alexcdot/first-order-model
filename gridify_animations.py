import os
import numpy as np
from moviepy.editor import *

fps = 12
time_length = 45
worker_id = int(os.environ['WORKER_ID'])
if os.environ.get('FPS') is not None:
    fps = int(os.environ['FPS'])
# max time
base_grid_size=(4, 8)

grid_sizes = [
    (4, 8),
    (4, 8),
    (4, 8),
    (5, 10),
]

downscale_factor = 2

video_width = 256
scaled_video_width = video_width // downscale_factor

target_resolutions = [(grid_size[0] * scaled_video_width,
                       grid_size[1] * scaled_video_width)
                      for grid_size in grid_sizes]

classes = ['frosh', 'smore', 'junior', 'senior']
my_class = classes[worker_id % 4]

data_dir = 'data'

time_id = str((worker_id % 4) * time_length) + '_' + str((worker_id % 4) * time_length + time_length)
result_video_dir = os.path.join(data_dir, my_class + '_videos')
all_video_names = [video_name for video_name in os.listdir(result_video_dir)
                   if '.mp4' in video_name]
my_video_names = []
for video_name in all_video_names:
    detected_fps = int(video_name.split('_')[4])
    detected_time_id = '_'.join(video_name.split('_')[2:4])
    if fps == detected_fps and time_id == detected_time_id:
        my_video_names.append(video_name)

"""
- Initialize big boy video
- Resize videos to (128, 128), then slot them in
- Pad them appropriately (so have horizontal offset)
- Get audio from official source
- Profit
"""

grid_size = grid_sizes[worker_id % 4]
target_resolution = target_resolutions[worker_id % 4]

movie_tensor = np.zeros((fps * time_length, target_resolution[0],
                  target_resolution[1], 3))

for i, video_name in enumerate(my_video_names):
    video_path = os.path.join(result_video_dir, video_name)
    clip = VideoFileClip(video_path)
    clip = clip.resize(width=clip.size[0] // downscale_factor)
    w, h = clip.size
    grid_y = i // grid_size[1]
    grid_x = i % grid_size[1]
    for j, frame in enumerate(clip.iter_frames()):
        if j == 0:
            print("frame shape:", frame.shape)
        if j >= fps * time_length:
            break
        y_start = grid_y * scaled_video_width
        x_start = grid_x * scaled_video_width
        movie_tensor[j, y_start:y_start + h,
                     x_start:x_start+w] = frame

numpy_backup_name = 'movie_tensor_' + str(worker_id % 4) + '.npy'

movie_name = 'still_alive_movie_' + str(worker_id % 4) + '.mp4'

# with open(os.path.join(data_dir, numpy_backup_name), 'wb+') as f:
#     np.save(f, movie_tensor)

movie = ImageSequenceClip(list(movie_tensor), fps=fps)
cropped_audio = AudioFileClip(os.path.join(
    data_dir, 'still_alive_good_cropped.ogg'
))

print('durations:', movie.duration, cropped_audio.duration)

cropped_audio = cropped_audio.subclip(t_start=(worker_id % 4) * time_length,
    t_end=min((worker_id % 4) * time_length + time_length, cropped_audio.duration))
movie = movie.set_audio(cropped_audio)
movie = movie.resize(width=base_grid_size[1] * scaled_video_width)
print(movie_name)
movie.write_videofile(os.path.join(data_dir, movie_name), audio_codec='aac')
    