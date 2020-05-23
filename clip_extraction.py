from moviepy.editor import *
import os

ice_clip = VideoFileClip('data/trump_ice.webm')
print('dims:', ice_clip.get_frame(0).shape)

def create_crop_func(y1, y2, x1, x2):
    def crop(image):
        return image[y1:y2, x1:x2,:]
    
    return crop

#336 x 596
crop_func = create_crop_func(10, 150, 230, 370)

cropped_ice = ice_clip.fl_image(crop_func).subclip(0, 6.5)
cropped_ice.write_videofile('data/trump_cropped_ice.mp4')