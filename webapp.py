# import the necessary packages

import argparse
import logging
import logging.handlers
import queue
import sys
import threading
import time
import urllib.request
# import time
# import threading
from multiprocessing import Process
from pathlib import Path
from typing import List, NamedTuple

import cv2
import cvlib as cv
import dlib
import imutils
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from aiortc.contrib.media import MediaPlayer
from cvlib.object_detection import draw_bbox
from imutils.video import FPS
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from streamlit_webrtc import (ClientSettings, VideoTransformerBase, WebRtcMode,
                              webrtc_streamer)

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av

# col1, col2 = st.beta_columns([1,2])
# with col1:
#     st.sidebar.image('logo.png', width=100)
# with col2:
#     st.sidebar.write('7P1')
# st.sidebar.markdown('<h1 style="float: left;">7P1</h1><img style="float: left;" src="./logo.png" />', unsafe_allow_html=True)
st.sidebar.title('ICU-7P1')
st.sidebar.image('logo.png', width=80)
# st.sidebar.markdown('<p style="text-align: center;">Centered text</p><img style="text-align: center;" src="logo.png" >', unsafe_allow_html=True)
html = """
  <style>
    /* Disable overlay (fullscreen mode) buttons */
  
    .stButton>button {
    # color: #4F8BF9;
    border-radius: 10%
    height: 4em;
    width: 19em;
}
    

  </style>
"""
st.markdown(html, unsafe_allow_html=True)
i = st.sidebar.button("Introduction")
add_selectbox = st.sidebar.selectbox(
  

    'What to do?',
    ('<SELECT>','Image Mask','OpenCV Vs Masking','Video Frame Count','Person Detection(Image)','Person Detection(Video)', 
    'Real Time Simulation', 'Real Time Object Detection','Customer Count')
)

###################################################################

if i:
    add_selectbox = '<SELECT>'
    st.title("Psifiólexi (7P1)")
    st.header("About Our Work")
    st.video("ICU_final.mp4")
    col3, col4 = st.beta_columns(2)
    col3.subheader("Computer Vision")
    col3.markdown('Here, the input image is converted into black and white. This makes it super easy for computers to make a prediction on what the image contains. It basically compares the white pixels to the black pixels, to form a simplified version of the image')
    col4.subheader("Visual")
    col4.image('CV.png', use_column_width=True)
  
 
   
######################################################

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)

import os

DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

logging.basicConfig(
    format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
    "%(message)s",
    force=True,
)

logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

st_webrtc_logger = logging.getLogger("streamlit_webrtc")
st_webrtc_logger.setLevel(logging.DEBUG)

fsevents_logger = logging.getLogger("fsevents")
fsevents_logger.setLevel(logging.WARNING)


#####################################################

def trackbar():
	h_min = st.slider('Hue min',0,179,0)
	h_max = st.slider('Hue max',0,179,179)
	s_min = st.slider('Sat min',0,255,0)
	s_max = st.slider('Sat max',0,255,255)
	v_min = st.slider('Val min',0,255,156)
	v_max = st.slider('Val max',0,255,255)
	return h_min,h_max,s_min,s_max,v_min,v_max 

###############################

# multi image stacker
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
######################################################################################################################################################


object_detection_page = "Real time object detection "



logger.debug("=== Alive threads ===")
for thread in threading.enumerate():
    if thread.is_alive():
        logger.debug(f"  {thread.name} ({thread.ident})")

    ##########################################################################################
########################################################
if add_selectbox == 'Person Detection(Image)':
    st.header('Object Detection on Image')
    selected_metrics = st.selectbox(
        label="Choose Image...", options=['SuperMarket-image-1','SuperMarket-image-2']
    )
    # im = cv2.imread('sp.jpg')
    if selected_metrics == 'SuperMarket-image-1':
        im = cv2.imread('sp.jpg')
    if selected_metrics == 'SuperMarket-image-2':
        im = cv2.imread('sp2.jpg')

    col3, col4 = st.beta_columns(2)
    col3.subheader("Original")
    col3.image(im, use_column_width=True)
    bbox, label, conf = cv.detect_common_objects(im)
    output_image = draw_bbox(im, bbox, label, conf)
    col4.subheader("Output")
    col4.image(output_image, use_column_width=True)

    st.write("Total people count : ",str(label.count('person')))
    # print('Number of cars in the image is '+ str(label.count('person')))
    cv2.destroyAllWindows()






#############################################################################

# construct the argument parse and parse the arguments
if add_selectbox == 'Video Frame Count':
    st.header('Caluclate frame rate and convert to grayscale')
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-v", "--video", required=True,
    # 	help="path to input video file")
    # args = vars(ap.parse_args())
    # open a pointer to the video stream and start the FPS timer
    col1, col2 = st.beta_columns(2)
    col1.subheader("Original video")
    image_placeholder_1 = col1.empty()
    stream = cv2.VideoCapture("CCTV.mp4")
    # col1.video("CCTV.mp4")
    if True:
    
        video = cv2.VideoCapture('CCTV.mp4')
        
        while True:
            success, image = video.read()
            if not success:
                break
            image_placeholder_1.image(image, channels="BGR")
    video.release()
    cv2.destroyAllWindows()
    fps = FPS().start()
    col2.subheader("Video processing...")
    frame_st = col2.empty()
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video file stream
        (grabbed, frame) = stream.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        # resize the frame and convert it to grayscale (while still
        # retaining 3 channels)
        frame = imutils.resize(frame, width=450)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.dstack([frame, frame, frame])
        # display a piece of text to the frame (so we can benchmark
        # fairly against the fast method)
        cv2.putText(frame, "Processed Video", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
        # show the frame and update the FPS counter
        # cv2.imshow("Frame", frame)
        frame_st.image(frame)
        cv2.waitKey(1)
        fps.update()

        # stop the timer and display FPS information
    fps.stop()
    # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    st.write("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    st.write("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    stream.release()

    cv2.destroyAllWindows()

##########################################################

if add_selectbox == 'Real Time Simulation':
    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.threshold1 = 100
            self.threshold2 = 200

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            img = cv2.cvtColor(
                cv2.Canny(img, self.threshold1, self.threshold2), cv2.COLOR_GRAY2BGR
            )

            return img


    ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    if ctx.video_transformer:
        ctx.video_transformer.threshold1 = st.slider("Threshold1", 0, 1000, 100)
        ctx.video_transformer.threshold2 = st.slider("Threshold2", 0, 1000, 200)


#################################################################################################

if add_selectbox == 'Real Time Object Detection':
    """Object detection demo with MobileNet SSD.
    """
    MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
    MODEL_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.caffemodel"
    PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
    PROTOTXT_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.prototxt.txt"

    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class Detection(NamedTuple):
        name: str
        prob: float

    class MobileNetSSDVideoTransformer(VideoTransformerBase):
        confidence_threshold: float
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            self._net = cv2.dnn.readNetFromCaffe(
                str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
            )
            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self.result_queue = queue.Queue()

        def _annotate_image(self, image, detections):
            # loop over the detections
            (h, w) = image.shape[:2]
            result: List[Detection] = []
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence_threshold:
                    # extract the index of the class label from the `detections`,
                    # then compute the (x, y)-coordinates of the bounding box for
                    # the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    name = CLASSES[idx]
                    result.append(Detection(name=name, prob=float(confidence)))

                    # display the prediction
                    label = f"{name}: {round(confidence * 100, 2)}%"
                    cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        image,
                        label,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[idx],
                        2,
                    )
            return image, result

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            image = frame.to_ndarray(format="bgr24")
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
            )
            self._net.setInput(blob)
            detections = self._net.forward()
            annotated_image, result = self._annotate_image(image, detections)

            # NOTE: This `transform` method is called in another thread,
            # so it must be thread-safe.
            self.result_queue.put(result)

            return annotated_image

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=MobileNetSSDVideoTransformer,
        async_transform=True,
    )

    # confidence_threshold = st.slider(
    #     "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    # )
    # if webrtc_ctx.video_transformer:
    #     webrtc_ctx.video_transformer.confidence_threshold = confidence_threshold

    if st.checkbox("Show the detected labels", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            while True:
                if webrtc_ctx.video_transformer:
                    try:
                        result = webrtc_ctx.video_transformer.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break

 


if add_selectbox == 'Customer Count':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", default='mobilenet_ssd/MobileNetSSD_deploy.prototxt',
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", default='mobilenet_ssd/MobileNetSSD_deploy.caffemodel',
        help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str,default = 'videos/example_01.mp4',
        help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str, default = 'output/output_01.avi',
        help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
        help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
        help="# of skip frames between detections")
    args = vars(ap.parse_args())

    # MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
    MODEL_LOCAL_PATH = "./models/MobileNetSSD_deploy.caffemodel"
    # PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
    PROTOTXT_LOCAL_PATH ="./models/MobileNetSSD_deploy.prototxt.txt"

    # initialize the list of class labels MobileNet SSD was trained to
    # detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))

    # if a video path was not supplied, grab a reference to the webcam
    if not args.get("input", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

    # otherwise, grab a reference to the video file
    else:
        print("[INFO] opening video file...")
        vs = cv2.VideoCapture(args["input"])

    # initialize the video writer (we'll instantiate later if need be)
    writer = None

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    # start the frames per second throughput estimator
    fps = FPS().start()
    st.title("Real Time Object Tracking")
    st.subheader("Counting the no. of person")
    frame_st = st.empty()
    # loop over frames from the video stream
    while True:
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame

        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if args["input"] is not None and frame is None:
            break

        
        # resize the frame to have a maximum width of 500 pixels (the
        # less data we have, the faster we can process it), then convert
        # the frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                (W, H), True)

        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % args["skip_frames"] == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []

            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > args["confidence"]:
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a person, ignore it
                    if CLASSES[idx] != "person":
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(frame, (0, 3*H // 4), (W, 3* H // 4), (0, 255, 255), 2)
        cv2.putText(frame, "GATE", (0,3* H // 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) ############################@##########

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        to.counted = True

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("OUT", totalUp),
            ("IN", totalDown),
            ("Status", status),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)

        # show the output frame
        frame_st.image(frame)

        # frame_st.image(frame)
        key = cv2.waitKey(1) & 0xFF
        

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    # if we are not using a video file, stop the camera video stream
    if not args.get("input", False):
        vs.stop()

    # otherwise, release the video file pointer
    else:
        vs.release()

    # close any open windows
    cv2.destroyAllWindows()
##################################################################################################################

if add_selectbox == 'Person Detection(Video)':
    # cap = cv2.VideoCapture('p_sp.avi')
    # image_placeholder_1 = st.empty()
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     image_placeholder_1.image(frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()
    def org_video(image_placeholder_1):
            
        if True:
            video = cv2.VideoCapture('CCTV.mp4')
            
            while True:
                success, image = video.read()
                if not success:
                    break
                image_placeholder_1.image(image, channels="BGR")
        video.release()
        cv2.destroyAllWindows()

    def process_video(image_placeholder_2):
            
        if True:
            video2 = cv2.VideoCapture('p_sp_1.avi')
            
            while True:
                ret, image2 = video2.read()
                if not ret:
                    break
                image_placeholder_2.image(image2)
        video2.release()
        cv2.destroyAllWindows()
    
    st.header("Object Detection on Video")
    col3, col4 = st.beta_columns(2)
    col3.subheader("Original")
    col4.subheader("Output")
    image_placeholder_1 = col3.empty()   
    image_placeholder_2 = col4.empty()
    # t_add = threading.Thread(target = org_video(image_placeholder_1))
    # t_del = threading.Thread(target = process_video(image_placeholder_2))
    # t_add.start()
    # t_del.start()                  
    p1 = Process(target = org_video(image_placeholder_1))
    p1.start()
    p2 = Process(target = process_video(image_placeholder_2))
    p2.start()                                                
    


##################################################################################################################

if add_selectbox == 'OpenCV Vs Masking':

    st.title("Comparing with openCV ")
    

    # '''HSV - Hue, Saturation, Value
    # HSV may also be called HSB (short for hue, saturation and brightness).
    # hue max -> 360 (opencv supports till 180 vals(0-179))'''

    def empty(a):
        pass


    def HSV(img):
        # TrackBar()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        
        h_min,h_max,s_min,s_max,v_min,v_max = trackbar()
        
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(img_hsv, lower, upper)
        img_result = cv2.bitwise_and(img,img,mask=mask)
        return mask


    def getCountours(img_h, img_c):
        contours, hierarchy = cv2.findContours(img_c, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            cv2.drawContours(img_contour1, cnt, -1, (0,255,0),3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img_detect1,(x,y),(x+w,y+h),(255,0,0),2)
        
        contours, hierarchy = cv2.findContours(img_h, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            cv2.drawContours(img_contour2, cnt, -1, (0,255,0),3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img_detect2,(x,y),(x+w,y+h),(255,0,0),2)

    def putLabel(img, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (10,400), font, 1,(255,255,255),2,cv2.LINE_AA)
            
    img = cv2.imread('pinwheel2.png')
    img_contour1 = img.copy()
    img_contour2 = img.copy()
    img_detect1 = img.copy()
    img_detect2 = img.copy()
    img_hsv = HSV(img)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(7,7),1)
    img_canny = cv2.Canny(img,300,300)

    getCountours(img_hsv, img_canny)

    putLabel(img, 'Original')
    putLabel(img_gray, 'Gray Scaling')
    putLabel(img_blur, 'Blured Image')
    putLabel(img_canny, 'OpenCV mask')
    putLabel(img_hsv, 'Our mask')
    putLabel(img_contour1, 'OpenCV contour')
    putLabel(img_contour2, 'Our contour')
    putLabel(img_detect1,'OpenCV detected')
    putLabel(img_detect2,'We detected')

    img_ver = stackImages(0.60,([img,img_gray,img_blur],
                                [img_canny, img_contour1,img_detect1],
                                [img_hsv, img_contour2, img_detect2]))
    # cv2.imshow('Output',img_ver)
    st.image(img_ver)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    ################################################################################
if add_selectbox == 'Image Mask':
    def empty(a):
        pass
    while True:
        img = cv2.imread('pinwheel2.png')
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min,h_max,s_min,s_max,v_min,v_max = trackbar()
        
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(img_hsv, lower, upper)
        img_result = cv2.bitwise_and(img,img,mask=mask)
        img_ver = stackImages(0.63,([img,img_hsv],[mask,img_result]))
        st.image(img_ver)
        cv2.waitKey(1)


    cv2.destroyAllWindows()
