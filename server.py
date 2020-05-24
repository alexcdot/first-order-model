import matplotlib
matplotlib.use('Agg')
import os, sys
import random
from filelock import FileLock

# Make sure the processes use different GPUs

worker_id_file = "current_worker_id.txt"
worker_id_lock_file = "current_worker_id.txt.lock"

num_processes = 8
current_worker_id = random.randint(0, num_processes - 1)

lock = FileLock(worker_id_lock_file, timeout=10)

with lock:
    if os.path.exists(worker_id_file):
        with open(worker_id_file, 'r') as f:
            lines = f.read().split()
            if len(lines) > 0:
                try:
                    current_worker_id = (int(lines[-1]) + 1) % num_processes
                except Exception as e:
                    print('Error in getting worker id:', e)
    with open(worker_id_file, 'a') as f:
        f.write('\n' + str(current_worker_id))

os.environ['CUDA_VISIBLE_DEVICES'] = str(current_worker_id)

is_static_only = int(current_worker_id) % 8 == 1

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
    url_for, send_from_directory, render_template, session, g
from werkzeug.utils import secure_filename
import time
from moviepy.editor import *
from moviepy.video.fx.all import crop
import ffmpeg
import uuid
import re

"""
visualize results for test image
"""

import matplotlib.pyplot as plt
import cv2
import sys
import tensorflow as tf
import face_recognition

from fer.model import predict, image_to_tensor, deepnn


UPLOAD_FOLDER = './server_data'
# Unnecessary, taken care of on frontend
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'webm'}


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

run_fer = (os.environ.get('RUN_FER') != '0')

if run_fer and not is_static_only:
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    show_box = True

    face_x = tf.placeholder(tf.float32, [None, 2304])
    y_conv = deepnn(face_x)
    probs = tf.nn.softmax(y_conv)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('fer/ckpt')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)     
    saver.restore(sess, ckpt.model_checkpoint_path)


def format_image(image):
    # print(image.shape, image.dtype)
    # image = image.astype('uint8') * 255
    recog_faces = face_recognition.face_locations(image, model='cnn')
    
    # top, right, bottom, left
    faces = []
    for (y1,x2,y2,x1) in recog_faces:
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        faces.append((x, y, w, h))
    # None is no face found in image
    if not len(faces) > 0:
        return None, None
    max_are_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face
    # face to image
    face_coor =  max_are_face
    # turn gray
    if len(image.shape) > 2 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    gray_image = gray_image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
    # Resize image to network size
    try:
        gray_image = cv2.resize(gray_image, (48, 48), interpolation=cv2.INTER_CUBIC)
    except Exception:
        print("[+} Problem during resize")
        return None, None
    return  gray_image, face_coor
    
    
def img_to_fer(frame):
    detected_face, face_coor = format_image(frame)
    if show_box and face_coor is not None:
        [x,y,w,h] = face_coor
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    if detected_face is None:
        print("No face spotted")
        return 0, 0
        
    tensor = image_to_tensor(detected_face)
    result = sess.run(probs, feed_dict={face_x: tensor})

    score = result[0]
    predicted = np.argmax(score)
    print("Image mean:", frame.mean())
    print("The Expression is %s" %str(class_names[int(predicted)]))
    return score, predicted


def video_to_fer(driving_video, max_frames=50):
    total_score = np.zeros(len(class_names))
    # unpack generators
    driving_video = [img for img in driving_video]
    print("VIDEO SIZE:", driving_video[0].shape)
    indices = list(range(len(driving_video)))
    
    if len(driving_video) > max_frames:
        indices = np.random.choice(len(driving_video), max_frames)
    
    for i in indices:
        img = driving_video[i]
        score, predicted = img_to_fer(img)
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
    '--source_image', 'data/robot_trump.jpg',
    '--result_video', 'result_robot_trump.mp4',
    '--checkpoint', 'vox-adv-cpk.pth.tar',
    '--adapt_scale', '--relative'])

if not is_static_only:
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

app = Flask(__name__)
app.config.from_mapping(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    SECRET_KEY='dev-swag-yolo',
    SEND_FILE_MAX_AGE_DEFAULT=0
)
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
    return '.' in filename
    # return '.' in filename and \
    #        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.before_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = session.get('reporter_status')
        g.group_name = session.get('group_name')
        g.submission_link = session.get('submission_link')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

# https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image/39382475
def crop_square_center(img):
    y, x = img.shape[:2]
    if y == x:
        return img

    if y < x:
        startx = x//2-(y//2)
        return img[:,startx:startx+y]
      
    if x < y:
        starty = y//2-(x//2)
        return img[starty:starty+x,:]


def adjust_clip(driving_video_path, fps, t, target_resolution):
    sparser_driving_video = '.'.join(driving_video_path.split('.')[:-1]) + '_sparse.mp4'
    (
        ffmpeg
        .input(driving_video_path)
        .output(sparser_driving_video, r=fps, t=t, acodec="aac", vcodec="libx264")
        .overwrite_output()
        .run()
    )

    orig_clip = VideoFileClip(sparser_driving_video)
    (w, h) = orig_clip.size
    new_dim = min(w, h)
    print("og size:", w, h)

    cropped_clip = crop(orig_clip, width=new_dim, height=new_dim, x_center=w/2, y_center=h/2)
    resized_clip = cropped_clip.resize(width=target_resolution[0])
    return resized_clip

    
def run_cloning(source_image_path, driving_video_path, result_video_path=None,
                run_fer=run_fer, fps=12, t=30, resized_clip=None):
    global generator, kp_detector
    target_resolution = (256, 256)

    source_image = imageio.imread(source_image_path)
    source_image = crop_square_center(source_image)
    
    if resized_clip is None:
        sparser_driving_video = '.'.join(driving_video_path.split('.')[:-1]) \
            + '_' + str(int(time.time())) + '_sparse.mp4'
        (
            ffmpeg
            .input(driving_video_path)
            .output(sparser_driving_video, r=fps, t=t, acodec="aac", vcodec="libx264")
            .overwrite_output()
            .run()
        )

        orig_clip = VideoFileClip(sparser_driving_video)
        (w, h) = orig_clip.size
        new_dim = min(w, h)
        print("og size:", w, h)

        cropped_clip = crop(orig_clip, width=new_dim, height=new_dim, x_center=w/2, y_center=h/2)
        resized_clip = cropped_clip.resize(width=target_resolution[0])

    print("resized clip", resized_clip.size)
    resized_clip = resized_clip.set_fps(fps)
    # resized_clip.write_videofile('resizing_test.mp4')

    orig_audio = resized_clip.audio
    driving_video = resized_clip.iter_frames()

    if run_fer:
        classes_by_score = video_to_fer(driving_video)

    # reset driving video
    driving_video = resized_clip.iter_frames()
    source_image = resize(source_image, target_resolution)[..., :3]
    driving_video = [resize(frame, target_resolution)[..., :3] for frame in driving_video]

    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    
    if result_video_path is None:
        output_filename = 'mind_theft_' + str(int(time.time()))
        if run_fer:
            top3_classes = '_'.join(list(classes_by_score.keys())[:3])
            output_filename += '_' + top3_classes

        output_filename += '.mp4'
        result_video_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    else:
        output_filename = result_video_path.split('/')[-1]

    # print('predictions shape', len(predictions), predictions[0].shape, predictions[0].dtype)
    new_video = ImageSequenceClip([img_as_ubyte(frame) for frame in predictions], fps=fps)
    # H x W x 3 format
    new_video = new_video.set_audio(orig_audio)
    new_video.write_videofile(result_video_path, audio_codec='aac')

    return output_filename


reporter_needs = {
    'john roberts': {'Happy'},
    'jim acoasta': {'Angry', 'Fear', 'Disgust'}
}


@app.route('/generate-fake', methods=['GET', 'POST'])
def generate_fake():
    # Just redirect.
    return redirect('/')

@app.route('/viva-la-revolution', methods=['GET'])
def revolution():
    return render_template('revolution.html')

general_methods = ['GET']
if not is_static_only:
    general_methods += ['POST']

@app.route('/', methods=general_methods)
def general():
    if request.method == 'POST':
        if len(request.form) > 0 and request.form.get('reporter_name'):
            req_data = request.form
            print(req_data)
            response_link = req_data.get('response_link')
            reporter_name = req_data.get('reporter_name')
            
            if 'user_id' not in session:
                session['user_id'] = str(uuid.uuid4())
            
            if 'reporter_status' not in session:
                session['reporter_status'] = {}

            # Set it to false by default
            if reporter_name is not None and reporter_name in reporter_needs:
                session.modified = True
                if reporter_name not in session['reporter_status'] or \
                   type(session['reporter_status'][reporter_name]) is bool:
                    session['reporter_status'][reporter_name] = {
                        'is_convinced': False,
                        'response_link': None,
                    }
                session['reporter_status'][reporter_name]['is_convinced'] = False

                if response_link.split('.')[-1] == 'mp4' and \
                   reporter_name in reporter_needs and \
                   os.path.exists(
                      os.path.join(app.config['UPLOAD_FOLDER'], response_link.split('/')[-1])
                   ):
                    key_emotions = response_link.split('.')[-2].split('_')[-3:]
                    session['reporter_status'][reporter_name]['response_link'] = response_link
                    
                    print(key_emotions, reporter_needs[reporter_name])
                    for emotion in key_emotions:
                        if emotion in reporter_needs[reporter_name]:
                            session['reporter_status'][reporter_name]['is_convinced'] = True
                            break

            print("Session:", session)
            return render_template('base_client.html')
        
        global generator, kp_detector
        
        req_data = request.files
        source_image = req_data.get('source_image')
        driving_video = req_data.get('driving_video')
        broadcast_video = req_data.get('broadcast_video')
        concat_files = req_data.getlist('concat_files')
        
        form_data = request.form
        headline = form_data.get('headline')
        group_name = form_data.get('group_name')

        # Submit broadcast video
        if broadcast_video is not None and broadcast_video.filename is not None:
            print("headline, group id and broadcast video", headline, group_name, broadcast_video)
            video_name = 'breaking_news_'
            if headline:
                headline = secure_filename(headline)
                video_name += headline + '_'
                
            if group_name:
                group_name = secure_filename(group_name)
                video_name += group_name + '_'

            video_name += str(int(time.time())) + '.mp4'
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
            broadcast_video.save(video_path)
            
            # Make this a new template site? Or let it unlock something?
            base_url = 'https://clone.verafy.me/uploads/'
            session['submission_link'] = os.path.join(
                base_url, video_name
            )
            session['group_name'] = group_name
            session.modified = True
            return redirect('/viva-la-revolution')
            # redirect(url_for('uploaded_file', filename=video_name))
        
        # Run concatenation of video files
        
        if len(concat_files) > 1:
            print('files to concatenate:', concat_files)
            valid_paths = []
            valid_in_files = []

            # filter and save files
            for file in concat_files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    video_path = os.path.join('user_data', 'concat_' + filename)
                    file.save(video_path)
                    valid_paths.append(video_path)
                    in_file = ffmpeg.input(video_path)
                    # This stuff didn't work
                    # example: scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2
                    # scaled_video = in_file.video.filter(
                    #    'scale', '256:256', 'force_original_aspect_ratio=decrease,pad=256:256:(ow-iw)/2:(oh-ih)/2'
                    # )
                    valid_in_files.extend([in_file.video, in_file.audio])
            
            # paths_file = os.path.join('user_data', 'concat_' + str(int(time.time())) + '.txt')
            # with open(paths_file, 'w+') as f:
            #    for path in valid_paths:
            #        valid_in_files.
            #        f.write("file '" + path + "'\n")
            concatenated_name = 'breaking_news_' + str(int(time.time())) + '.mp4' 
            concatenated_path = os.path.join(
                app.config['UPLOAD_FOLDER'], concatenated_name
            )
            (
                ffmpeg
                .concat(*valid_in_files, v=1, a=1)
                .output(concatenated_path, r=12, acodec="aac", vcodec="libx264")
                .run()
            )
            return redirect(url_for('uploaded_file',
                                    filename=concatenated_name))

        print('source image:', source_image)
        print('driving video:', driving_video)
        
        default_output_filename = 'fake_result_1590191838_Sad_Fear_Neutral.mp4'
        default_output_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                           default_output_filename)
        if (source_image is None or source_image.filename is None or len(source_image.filename) == 0) and \
           (driving_video is None or driving_video.filename is None or len(driving_video.filename) == 0) and \
           os.path.exists(default_output_path):
            return redirect(url_for('uploaded_file',
                                filename=default_output_filename))

        source_image_path = opt.source_image
        driving_video_path = opt.driving_video
        
        if source_image and allowed_file(source_image.filename):
            filename = secure_filename(source_image.filename)
            source_image_path = os.path.join('user_data', filename)
            source_image.save(source_image_path)
        
        if driving_video and allowed_file(driving_video.filename):
            filename = secure_filename(driving_video.filename)
            driving_video_path = os.path.join('user_data', filename)
            driving_video.save(driving_video_path)
        
        output_filename = run_cloning(source_image_path, driving_video_path)
        return redirect(url_for('uploaded_file',
                                filename=output_filename))

    return render_template('base_client.html')

if __name__ == '__main__':
    app.run(debug=False, port=4000, use_reloader=False, host='0.0.0.0') #run app in debug mode on port 5000
