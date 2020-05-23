import os
from moviepy.editor import *

data_dir = 'data'

movies_list = [
    'still_alive_movie_0.mp4',
    'still_alive_movie_1.mp4',
    'still_alive_movie_2.mp4',
    'still_alive_movie_3.mp4',
]

audio_clip = AudioFileClip(
    os.path.join(data_dir, 'still_alive_good_cropped_stylized.ogg')
)

video_clips = [
    VideoFileClip(os.path.join(data_dir, movie_name))
    for movie_name in movies_list
]
# correction from Aperture Science to Project Hyperskelion
audio_clip = audio_clip.subclip(t_end=video_clips[0].duration)

video_clips[0] = video_clips[0].set_audio(audio_clip)
video_clips[3] = video_clips[3].subclip(t_end=42)

final_clip = concatenate_videoclips(video_clips)

final_clip.write_videofile(
    os.path.join(data_dir, 'still_alive_movie_all.mp4')
)