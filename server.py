import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull

import binascii
import base64
import io
from flask_cors import CORS
import json
from flask import Flask, flash, request, Response, redirect, \
    url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
import time
from moviepy.editor import *
from moviepy.video.fx.all import crop
import ffmpeg

"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

import fer_pytorch.transforms as transforms
from skimage import io
from skimage.transform import resize
from fer_pytorch.models import *


UPLOAD_FOLDER = './server_data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'webm'}


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
net = VGG('VGG19')
checkpoint = torch.load(os.path.join('fer_pytorch', 'FER2013_VGG19', 'PrivateTest_model.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def img_to_fer(raw_img):
    # raw_img = io.imread('fer_pytorch/images/1.jpg')
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)


    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    global net
    outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    score = F.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)

    print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))
    return score, predicted


def video_to_fer(driving_video):
    total_score = None
    
    for img in driving_video:
        score, predicted = img_to_fer(img)
        if total_score is None:
            total_score = score
        else:
            total_score += score
    
    sorted_classes = sorted(list(zip(class_names, total_score)), key = lambda x: x[1], reverse=True)
    classes_by_score = {fer_class: score for fer_class, score in sorted_classes}
    print(classes_by_score)
    return classes_by_score
  

parser = ArgumentParser()
parser.add_argument("--config", required=True, help="path to config")
parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
parser.add_argument("--driving_video", default='sup-mat/source.png', help="path to driving video")
parser.add_argument("--result_video", default='result.mp4', help="path to output")

parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                    help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                    help="Set frame to start from.")

parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")


parser.set_defaults(relative=False)
parser.set_defaults(adapt_scale=False)
parser.set_defaults(cpu=False)

opt = parser.parse_args([
    '--config', 'config/vox-adv-256.yaml',
    '--driving_video', 'data/trump_cropped_ice.mp4',
    '--source_image', 'data/05.png',
    '--result_video', 'result_mona_trump_abs.mp4',
    '--checkpoint', 'vox-adv-cpk.pth.tar',
    '--adapt_scale', '--relative'])
generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/generate-fake', methods=['GET', 'POST'])
def generate_fake():
    if request.method == 'POST':
        
        global generator, kp_detector
        
        req_data = request.files
        # if request.is_json:
        #     req_data = request.get_json()
        # else:
        #     data = request.data
        #     req_data = json.loads(data)
        source_image = req_data.get('source_image')
        driving_video = req_data.get('driving_video')

        print(source_image, source_image.filename)
        print(driving_video, driving_video.filename)

        source_image_path = None
        driving_video_path = None
        
        if source_image and allowed_file(source_image.filename):
            filename = secure_filename(source_image.filename)
            source_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            source_image.save(source_image_path)
            opt.source_image = source_image_path
        
        if driving_video and allowed_file(driving_video.filename):
            filename = secure_filename(driving_video.filename)
            driving_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            driving_video.save(driving_video_path)
            opt.driving_video = driving_video_path
        
        default_output_filename = 'default_fake_result.mp4'
        default_output_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                           default_output_filename)
        if source_image_path is None and driving_video_path is None \
           and os.path.exists(default_output_path):
            return redirect(url_for('uploaded_file',
                                    filename=default_output_filename))

        source_image = imageio.imread(opt.source_image)
        fps = 12
        target_resolution = (256, 256)
        sparser_driving_video = '.'.join(opt.driving_video.split('.')[:-1]) + '_sparse.mp4'
        (
            ffmpeg
            .input(opt.driving_video)
            .output(sparser_driving_video, r=12)
            .overwrite_output()
            .run()
        )
        
        orig_clip = VideoFileClip(sparser_driving_video)
        (w, h) = orig_clip.size
        cropped_clip = crop(orig_clip, width=h, height=h, x_center=w/2, y_center=h/2)
        resized_clip = cropped_clip.resize(width=target_resolution[0])

        resized_clip = resized_clip.set_fps(fps)
        resized_clip.write_videofile('resizing_test.mp4')
        
        orig_audio = resized_clip.audio
        driving_video = resized_clip.iter_frames()

        source_image = resize(source_image, target_resolution)[..., :3]
        driving_video = [resize(frame, target_resolution)[..., :3] for frame in driving_video]

        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        
        classes_by_score = video_to_fer(driving_video)
        
        output_filename = 'fake_result_' + str(int(time.time())) + '_' + list(classes_by_score.keys())[0] + '.mp4'
        opt.result_video = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # print('predictions shape', len(predictions), predictions[0].shape, predictions[0].dtype)
        new_video = ImageSequenceClip([img_as_ubyte(frame) for frame in predictions], fps=fps)
        # H x W x 3 format
        new_video = new_video.set_audio(orig_audio)
        new_video.write_videofile(opt.result_video)
        
        # imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)

        return redirect(url_for('uploaded_file',
                                filename=output_filename))
        
#         if 'data:image/jpeg;base64,' in image:
#             image = image[len("data:image/jpeg;base64,"):]

#         img = asarray(Image.open(io.BytesIO(base64.b64decode(image))))

#         response = {'message': 'image received', 'embedding': [1,2,3]}
#         response_pickled = json.dumps(response)
#         return Response(response=response_pickled, status=200,
#                         mimetype="application/json")
    return render_template('base_client.html')

if __name__ == '__main__':
    app.run(debug=False, port=4000, use_reloader=False, host='0.0.0.0') #run app in debug mode on port 5000