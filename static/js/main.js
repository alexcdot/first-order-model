/*
Copyright 2017 Google Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

'use strict';

// This code is adapted from
// https://rawgit.com/Miguelao/demos/master/mediarecorder.html

var mediaSource = new MediaSource();
mediaSource.addEventListener('sourceopen', handleSourceOpen, false);
var mediaRecorder;
var recordedBlobs;
var sourceBuffer;

var gumVideo = document.querySelector('video#gum');
var recordedVideo = document.querySelector('video#recorded');

var toggleCameraButton = document.querySelector('button#toggleCamera');
var recordButton = document.querySelector('button#record');
var playButton = document.querySelector('button#play');
var downloadButton = document.querySelector('button#download');

var mimeTypePlayback = 'video/webm; codecs="vp8"';
var mimeTypeRecording = 'video/webm; codecs="vp9"';
var fileExtension = '.webm';
var mimeType = 'video/webm';

toggleCameraButton.onclick = toggleCamera;
recordButton.onclick = toggleRecording;
playButton.onclick = play;
downloadButton.onclick = download;

console.log(location.host);
// window.isSecureContext could be used for Chrome
var isSecureOrigin = location.protocol === 'https:' ||
  location.host.includes('localhost');
if (!isSecureOrigin) {
  alert('getUserMedia() must be run from a secure origin: HTTPS' +
    '\n\nWebcam recording will not work');
}

var constraints = {
  audio: true,
  video: true
};

navigator.mediaDevices.getUserMedia(
  constraints
).then(
  successCallback,
  errorCallback
);

function successCallback(stream) {
  console.log('getUserMedia() got stream: ', stream);
  window.stream = stream;
  gumVideo.srcObject = stream;
  recordButton.disabled = false;
}

function errorCallback(error) {
  console.log('navigator.getUserMedia error: ', error);
}

function toggleCamera() {
  if (window.stream.active) {
    console.log('stopping the stream');
    window.stream.getTracks().forEach(track => track.stop());
    recordButton.disabled = true;
  } else {
    console.log('starting the stream');
    navigator.mediaDevices.getUserMedia(
      constraints
    ).then(
      successCallback,
      errorCallback
    );
  }
}

function handleSourceOpen(event) {
  console.log('MediaSource opened');
  // sourceBuffer = mediaSource.addSourceBuffer('video/webm; codecs="vp8"');
  sourceBuffer = mediaSource.addSourceBuffer(mimeTypePlayback);
  console.log('Source buffer: ', sourceBuffer);
}

function handleDataAvailable(event) {
  if (event.data && event.data.size > 0) {
    recordedBlobs.push(event.data);
  }
}

function handleStop(event) {
  console.log('Recorder stopped: ', event);
}

function toggleRecording() {
  if (recordButton.textContent === 'Start Recording') {
    startRecording();
  } else {
    stopRecording();
    recordButton.textContent = 'Start Recording';
    playButton.disabled = false;
    downloadButton.disabled = false;
  }
}

// The nested try blocks will be simplified when Chrome 47 moves to Stable
function startRecording() {
  // var options = {mimeType: 'video/webm;codecs=vp9', bitsPerSecond: 100000};
  var options = {mimeType: mimeTypeRecording, bitsPerSecond: 1000000};
  recordedBlobs = [];
  try {
    console.log("Trying to start media recorder")
    mediaRecorder = new MediaRecorder(window.stream, options);
  } catch (e0) {
    console.log('Unable to create MediaRecorder with options Object: ', options, e0);
    try {
      options = {bitsPerSecond: 1000000};
      mediaRecorder = new MediaRecorder(window.stream, options);
    } catch (e1) {
      console.log('Unable to create MediaRecorder with options Object: ', options, e1);
      try {
        options = {mimeType: 'video/mp4', bitsPerSecond: 1000000};
        mediaRecorder = new MediaRecorder(window.stream, options);
      } catch (e2) {
        console.log('Unable to create MediaRecorder with options Object: ', options, e2); 
        try {
          options = {mimeType: 'video/webm; codecs="opus,vp8"', bitsPerSecond: 1000000};
          mediaRecorder = new MediaRecorder(window.stream, options);
        } catch (e3) {
          alert('MediaRecorder is not supported by this browser.');
          console.error('Exception while creating MediaRecorder:', e3);
          return;
        }
      }
    }
  }
  console.log('Created MediaRecorder', mediaRecorder, 'with options', options);
  recordButton.textContent = 'Stop Recording';
  playButton.disabled = true;
  downloadButton.disabled = true;
  mediaRecorder.onstop = handleStop;
  mediaRecorder.ondataavailable = handleDataAvailable;
  mediaRecorder.start(10); // collect 10ms of data
  console.log('MediaRecorder started', mediaRecorder);
}

function stopRecording() {
  mediaRecorder.stop();
  console.log('Recorded Blobs: ', recordedBlobs);
  recordedVideo.controls = true;
}

function play() {
  var superBuffer = new Blob(recordedBlobs, {type: mimeType});
  recordedVideo.src = window.URL.createObjectURL(superBuffer);
}

function download() {
  var blob = new Blob(recordedBlobs, {type: mimeType});
  var url = window.URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = 'driving_video_' + String(Date.now()) + fileExtension;
  document.body.appendChild(a);
  a.click();
  setTimeout(function() {
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }, 100);
}