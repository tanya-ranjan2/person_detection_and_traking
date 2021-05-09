
# # import streamlit as st

# # import cv2
# # from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# # # webrtc_streamer(key="example")


# # class VideoTransformer(VideoTransformerBase):
# #     def __init__(self):
# #         self.threshold1 = 100
# #         self.threshold2 = 200

# #     def transform(self, frame):
# #         img = frame.to_ndarray(format="bgr24")

# #         img = cv2.cvtColor(
# #             cv2.Canny(img, self.threshold1, self.threshold2), cv2.COLOR_GRAY2BGR
# #         )

# #         return img


# # ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# # if ctx.video_transformer:
# #     ctx.video_transformer.threshold1 = st.slider("Threshold1", 0, 1000, 100)
# #     ctx.video_transformer.threshold2 = st.slider("Threshold2", 0, 1000, 200)

# #################################################################################################


# # # import the necessary packages
# # import streamlit as st
# # import cv2
# # import matplotlib.pyplot as plt
# # import cvlib as cv
# # from cvlib.object_detection import draw_bbox
# # im = cv2.imread('traffic.png')
# # col3, col4 = st.beta_columns(2)
# # col3.subheader("Original")
# # col3.image(im, use_column_width=True)
# # bbox, label, conf = cv.detect_common_objects(im)
# # output_image = draw_bbox(im, bbox, label, conf)
# # col4.subheader("Output")
# # col4.image(output_image, use_column_width=True)

# # st.write("Total people count : ",str(label.count('car')))


# # #######################################################

# import logging
# import logging.handlers
# import queue
# import threading
# import urllib.request
# from pathlib import Path
# from typing import List, NamedTuple

# try:
#     from typing import Literal
# except ImportError:
#     from typing_extensions import Literal  # type: ignore

# import av
# import cv2
# import numpy as np
# import streamlit as st
# from aiortc.contrib.media import MediaPlayer

# from streamlit_webrtc import (
#     ClientSettings,
#     VideoTransformerBase,
#     WebRtcMode,
#     webrtc_streamer,
# )

# HERE = Path(__file__).parent

# logger = logging.getLogger(__name__)


# # This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
# def download_file(url, download_to: Path, expected_size=None):
#     # Don't download the file twice.
#     # (If possible, verify the download using the file length.)
#     if download_to.exists():
#         if expected_size:
#             if download_to.stat().st_size == expected_size:
#                 return
#         else:
#             st.info(f"{url} is already downloaded.")
#             if not st.button("Download again?"):
#                 return

#     download_to.parent.mkdir(parents=True, exist_ok=True)

#     # These are handles to two visual elements to animate.
#     weights_warning, progress_bar = None, None
#     try:
#         weights_warning = st.warning("Downloading %s..." % url)
#         progress_bar = st.progress(0)
#         with open(download_to, "wb") as output_file:
#             with urllib.request.urlopen(url) as response:
#                 length = int(response.info()["Content-Length"])
#                 counter = 0.0
#                 MEGABYTES = 2.0 ** 20.0
#                 while True:
#                     data = response.read(8192)
#                     if not data:
#                         break
#                     counter += len(data)
#                     output_file.write(data)

#                     # We perform animation by overwriting the elements.
#                     weights_warning.warning(
#                         "Downloading %s... (%6.2f/%6.2f MB)"
#                         % (url, counter / MEGABYTES, length / MEGABYTES)
#                     )
#                     progress_bar.progress(min(counter / length, 1.0))
#     # Finally, we remove these visual elements by calling .empty().
#     finally:
#         if weights_warning is not None:
#             weights_warning.empty()
#         if progress_bar is not None:
#             progress_bar.empty()


# WEBRTC_CLIENT_SETTINGS = ClientSettings(
#     rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#     media_stream_constraints={"video": True, "audio": True},
# )

# import os

# DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

# logging.basicConfig(
#     format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
#     "%(message)s",
#     force=True,
# )

# logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

# st_webrtc_logger = logging.getLogger("streamlit_webrtc")
# st_webrtc_logger.setLevel(logging.DEBUG)

# fsevents_logger = logging.getLogger("fsevents")
# fsevents_logger.setLevel(logging.WARNING)




# ###############################
# st.header("WebRTC demo")

# object_detection_page = "Real time object detection (sendrecv)"



# logger.debug("=== Alive threads ===")
# for thread in threading.enumerate():
#     if thread.is_alive():
#         logger.debug(f"  {thread.name} ({thread.ident})")

# ##########################################################################################
# """Object detection demo with MobileNet SSD.
# This model and code are based on
# https://github.com/robmarkcole/object-detection-app
# """
# MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
# MODEL_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.caffemodel"
# PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
# PROTOTXT_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.prototxt.txt"

# CLASSES = [
#     "background",
#     "aeroplane",
#     "bicycle",
#     "bird",
#     "boat",
#     "bottle",
#     "bus",
#     "car",
#     "cat",
#     "chair",
#     "cow",
#     "diningtable",
#     "dog",
#     "horse",
#     "motorbike",
#     "person",
#     "pottedplant",
#     "sheep",
#     "sofa",
#     "train",
#     "tvmonitor",
# ]
# COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
# download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

# DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# class Detection(NamedTuple):
#     name: str
#     prob: float

# class MobileNetSSDVideoTransformer(VideoTransformerBase):
#     confidence_threshold: float
#     result_queue: "queue.Queue[List[Detection]]"

#     def __init__(self) -> None:
#         self._net = cv2.dnn.readNetFromCaffe(
#             str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
#         )
#         self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
#         self.result_queue = queue.Queue()

#     def _annotate_image(self, image, detections):
#         # loop over the detections
#         (h, w) = image.shape[:2]
#         result: List[Detection] = []
#         for i in np.arange(0, detections.shape[2]):
#             confidence = detections[0, 0, i, 2]

#             if confidence > self.confidence_threshold:
#                 # extract the index of the class label from the `detections`,
#                 # then compute the (x, y)-coordinates of the bounding box for
#                 # the object
#                 idx = int(detections[0, 0, i, 1])
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")

#                 name = CLASSES[idx]
#                 result.append(Detection(name=name, prob=float(confidence)))

#                 # display the prediction
#                 label = f"{name}: {round(confidence * 100, 2)}%"
#                 cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
#                 y = startY - 15 if startY - 15 > 15 else startY + 15
#                 cv2.putText(
#                     image,
#                     label,
#                     (startX, y),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     COLORS[idx],
#                     2,
#                 )
#         return image, result

#     def transform(self, frame: av.VideoFrame) -> np.ndarray:
#         image = frame.to_ndarray(format="bgr24")
#         blob = cv2.dnn.blobFromImage(
#             cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
#         )
#         self._net.setInput(blob)
#         detections = self._net.forward()
#         annotated_image, result = self._annotate_image(image, detections)

#         # NOTE: This `transform` method is called in another thread,
#         # so it must be thread-safe.
#         self.result_queue.put(result)

#         return annotated_image

# webrtc_ctx = webrtc_streamer(
#     key="object-detection",
#     mode=WebRtcMode.SENDRECV,
#     client_settings=WEBRTC_CLIENT_SETTINGS,
#     video_transformer_factory=MobileNetSSDVideoTransformer,
#     async_transform=True,
# )

# confidence_threshold = st.slider(
#     "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
# )
# if webrtc_ctx.video_transformer:
#     webrtc_ctx.video_transformer.confidence_threshold = confidence_threshold

# if st.checkbox("Show the detected labels", value=True):
#     if webrtc_ctx.state.playing:
#         labels_placeholder = st.empty()
#         # NOTE: The video transformation with object detection and
#         # this loop displaying the result labels are running
#         # in different threads asynchronously.
#         # Then the rendered video frames and the labels displayed here
#         # are not strictly synchronized.
#         while True:
#             if webrtc_ctx.video_transformer:
#                 try:
#                     result = webrtc_ctx.video_transformer.result_queue.get(
#                         timeout=1.0
#                     )
#                 except queue.Empty:
#                     result = None
#                 labels_placeholder.table(result)
#             else:
#                 break

# st.markdown(
#     "This demo uses a model and code from "
#     "https://github.com/robmarkcole/object-detection-app. "
#     "Many thanks to the project."
# )


# #####################################################################
# # def main():
# #     st.header("WebRTC demo")

# #     object_detection_page = "Real time object detection (sendrecv)"
# #     video_filters_page = (
# #         "Real time video transform with simple OpenCV filters (sendrecv)"
# #     )
# #     streaming_page = (
# #         "Consuming media files on server-side and streaming it to browser (recvonly)"
# #     )
# #     sendonly_page = "WebRTC is sendonly and images are shown via st.image() (sendonly)"
# #     loopback_page = "Simple video loopback (sendrecv)"
# #     app_mode = st.sidebar.selectbox(
# #         "Choose the app mode",
# #         [
# #             object_detection_page,
# #             video_filters_page,
# #             streaming_page,
# #             sendonly_page,
# #             loopback_page,
# #         ],
# #     )
# #     st.subheader(app_mode)

# #     if app_mode == video_filters_page:
# #         app_video_filters()
# #     elif app_mode == object_detection_page:
# #         app_object_detection()
# #     elif app_mode == streaming_page:
# #         app_streaming()
# #     elif app_mode == sendonly_page:
# #         app_sendonly()
# #     elif app_mode == loopback_page:
# #         app_loopback()

# #     logger.debug("=== Alive threads ===")
# #     for thread in threading.enumerate():
# #         if thread.is_alive():
# #             logger.debug(f"  {thread.name} ({thread.ident})")


# # def app_loopback():
# #     """ Simple video loopback """
# #     webrtc_streamer(
# #         key="loopback",
# #         mode=WebRtcMode.SENDRECV,
# #         client_settings=WEBRTC_CLIENT_SETTINGS,
# #         video_transformer_factory=None,  # NoOp
# #     )


# # def app_video_filters():
# #     """ Video transforms with OpenCV """

# #     class OpenCVVideoTransformer(VideoTransformerBase):
# #         type: Literal["noop", "cartoon", "edges", "rotate"]

# #         def __init__(self) -> None:
# #             self.type = "noop"

# #         def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
# #             img = frame.to_ndarray(format="bgr24")

# #             if self.type == "noop":
# #                 pass
# #             elif self.type == "cartoon":
# #                 # prepare color
# #                 img_color = cv2.pyrDown(cv2.pyrDown(img))
# #                 for _ in range(6):
# #                     img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
# #                 img_color = cv2.pyrUp(cv2.pyrUp(img_color))

# #                 # prepare edges
# #                 img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# #                 img_edges = cv2.adaptiveThreshold(
# #                     cv2.medianBlur(img_edges, 7),
# #                     255,
# #                     cv2.ADAPTIVE_THRESH_MEAN_C,
# #                     cv2.THRESH_BINARY,
# #                     9,
# #                     2,
# #                 )
# #                 img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

# #                 # combine color and edges
# #                 img = cv2.bitwise_and(img_color, img_edges)
# #             elif self.type == "edges":
# #                 # perform edge detection
# #                 img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
# #             elif self.type == "rotate":
# #                 # rotate image
# #                 rows, cols, _ = img.shape
# #                 M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
# #                 img = cv2.warpAffine(img, M, (cols, rows))

# #             return img

# #     webrtc_ctx = webrtc_streamer(
# #         key="opencv-filter",
# #         mode=WebRtcMode.SENDRECV,
# #         client_settings=WEBRTC_CLIENT_SETTINGS,
# #         video_transformer_factory=OpenCVVideoTransformer,
# #         async_transform=True,
# #     )

# #     if webrtc_ctx.video_transformer:
# #         webrtc_ctx.video_transformer.type = st.radio(
# #             "Select transform type", ("noop", "cartoon", "edges", "rotate")
# #         )

# #     st.markdown(
# #         "This demo is based on "
# #         "https://github.com/aiortc/aiortc/blob/2362e6d1f0c730a0f8c387bbea76546775ad2fe8/examples/server/server.py#L34. "  # noqa: E501
# #         "Many thanks to the project."
# #     )


# # def app_object_detection():
# #     """Object detection demo with MobileNet SSD.
# #     This model and code are based on
# #     https://github.com/robmarkcole/object-detection-app
# #     """
# #     MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
# #     MODEL_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.caffemodel"
# #     PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
# #     PROTOTXT_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.prototxt.txt"

# #     CLASSES = [
# #         "background",
# #         "aeroplane",
# #         "bicycle",
# #         "bird",
# #         "boat",
# #         "bottle",
# #         "bus",
# #         "car",
# #         "cat",
# #         "chair",
# #         "cow",
# #         "diningtable",
# #         "dog",
# #         "horse",
# #         "motorbike",
# #         "person",
# #         "pottedplant",
# #         "sheep",
# #         "sofa",
# #         "train",
# #         "tvmonitor",
# #     ]
# #     COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# #     download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
# #     download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

# #     DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# #     class Detection(NamedTuple):
# #         name: str
# #         prob: float

# #     class MobileNetSSDVideoTransformer(VideoTransformerBase):
# #         confidence_threshold: float
# #         result_queue: "queue.Queue[List[Detection]]"

# #         def __init__(self) -> None:
# #             self._net = cv2.dnn.readNetFromCaffe(
# #                 str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
# #             )
# #             self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
# #             self.result_queue = queue.Queue()

# #         def _annotate_image(self, image, detections):
# #             # loop over the detections
# #             (h, w) = image.shape[:2]
# #             result: List[Detection] = []
# #             for i in np.arange(0, detections.shape[2]):
# #                 confidence = detections[0, 0, i, 2]

# #                 if confidence > self.confidence_threshold:
# #                     # extract the index of the class label from the `detections`,
# #                     # then compute the (x, y)-coordinates of the bounding box for
# #                     # the object
# #                     idx = int(detections[0, 0, i, 1])
# #                     box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# #                     (startX, startY, endX, endY) = box.astype("int")

# #                     name = CLASSES[idx]
# #                     result.append(Detection(name=name, prob=float(confidence)))

# #                     # display the prediction
# #                     label = f"{name}: {round(confidence * 100, 2)}%"
# #                     cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
# #                     y = startY - 15 if startY - 15 > 15 else startY + 15
# #                     cv2.putText(
# #                         image,
# #                         label,
# #                         (startX, y),
# #                         cv2.FONT_HERSHEY_SIMPLEX,
# #                         0.5,
# #                         COLORS[idx],
# #                         2,
# #                     )
# #             return image, result

# #         def transform(self, frame: av.VideoFrame) -> np.ndarray:
# #             image = frame.to_ndarray(format="bgr24")
# #             blob = cv2.dnn.blobFromImage(
# #                 cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
# #             )
# #             self._net.setInput(blob)
# #             detections = self._net.forward()
# #             annotated_image, result = self._annotate_image(image, detections)

# #             # NOTE: This `transform` method is called in another thread,
# #             # so it must be thread-safe.
# #             self.result_queue.put(result)

# #             return annotated_image

# #     webrtc_ctx = webrtc_streamer(
# #         key="object-detection",
# #         mode=WebRtcMode.SENDRECV,
# #         client_settings=WEBRTC_CLIENT_SETTINGS,
# #         video_transformer_factory=MobileNetSSDVideoTransformer,
# #         async_transform=True,
# #     )

# #     confidence_threshold = st.slider(
# #         "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
# #     )
# #     if webrtc_ctx.video_transformer:
# #         webrtc_ctx.video_transformer.confidence_threshold = confidence_threshold

# #     if st.checkbox("Show the detected labels", value=True):
# #         if webrtc_ctx.state.playing:
# #             labels_placeholder = st.empty()
# #             # NOTE: The video transformation with object detection and
# #             # this loop displaying the result labels are running
# #             # in different threads asynchronously.
# #             # Then the rendered video frames and the labels displayed here
# #             # are not strictly synchronized.
# #             while True:
# #                 if webrtc_ctx.video_transformer:
# #                     try:
# #                         result = webrtc_ctx.video_transformer.result_queue.get(
# #                             timeout=1.0
# #                         )
# #                     except queue.Empty:
# #                         result = None
# #                     labels_placeholder.table(result)
# #                 else:
# #                     break

# #     st.markdown(
# #         "This demo uses a model and code from "
# #         "https://github.com/robmarkcole/object-detection-app. "
# #         "Many thanks to the project."
# #     )


# # def app_streaming():
# #     """ Media streamings """
# #     MEDIAFILES = {
# #         "big_buck_bunny_720p_2mb.mp4": {
# #             "url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_2mb.mp4",  # noqa: E501
# #             "local_file_path": HERE / "data/big_buck_bunny_720p_2mb.mp4",
# #             "type": "video",
# #         },
# #         "big_buck_bunny_720p_10mb.mp4": {
# #             "url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_10mb.mp4",  # noqa: E501
# #             "local_file_path": HERE / "data/big_buck_bunny_720p_10mb.mp4",
# #             "type": "video",
# #         },
# #         "file_example_MP3_700KB.mp3": {
# #             "url": "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_700KB.mp3",  # noqa: E501
# #             "local_file_path": HERE / "data/file_example_MP3_700KB.mp3",
# #             "type": "audio",
# #         },
# #         "file_example_MP3_5MG.mp3": {
# #             "url": "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_5MG.mp3",  # noqa: E501
# #             "local_file_path": HERE / "data/file_example_MP3_5MG.mp3",
# #             "type": "audio",
# #         },
# #     }
# #     media_file_label = st.radio(
# #         "Select a media file to stream", tuple(MEDIAFILES.keys())
# #     )
# #     media_file_info = MEDIAFILES[media_file_label]
# #     download_file(media_file_info["url"], media_file_info["local_file_path"])

# #     def create_player():
# #         return MediaPlayer(str(media_file_info["local_file_path"]))

# #         # NOTE: To stream the video from webcam, use the code below.
# #         # return MediaPlayer(
# #         #     "1:none",
# #         #     format="avfoundation",
# #         #     options={"framerate": "30", "video_size": "1280x720"},
# #         # )

# #     WEBRTC_CLIENT_SETTINGS.update(
# #         {
# #             "media_stream_constraints": {
# #                 "video": media_file_info["type"] == "video",
# #                 "audio": media_file_info["type"] == "audio",
# #             }
# #         }
# #     )

# #     webrtc_streamer(
# #         key=f"media-streaming-{media_file_label}",
# #         mode=WebRtcMode.RECVONLY,
# #         client_settings=WEBRTC_CLIENT_SETTINGS,
# #         player_factory=create_player,
# #     )


# # def app_sendonly():
# #     """A sample to use WebRTC in sendonly mode to transfer frames
# #     from the browser to the server and to render frames via `st.image`."""
# #     webrtc_ctx = webrtc_streamer(
# #         key="loopback",
# #         mode=WebRtcMode.SENDONLY,
# #         client_settings=WEBRTC_CLIENT_SETTINGS,
# #     )

# #     if webrtc_ctx.video_receiver:
# #         image_loc = st.empty()
# #         while True:
# #             try:
# #                 frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
# #             except queue.Empty:
# #                 print("Queue is empty. Stop the loop.")
# #                 webrtc_ctx.video_receiver.stop()
# #                 break

# #             img_rgb = frame.to_ndarray(format="rgb24")
# #             image_loc.image(img_rgb)


# # if __name__ == "__main__":
# #     import os

# #     DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

# #     logging.basicConfig(
# #         format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
# #         "%(message)s",
# #         force=True,
# #     )

# #     logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

# #     st_webrtc_logger = logging.getLogger("streamlit_webrtc")
# #     st_webrtc_logger.setLevel(logging.DEBUG)

# #     fsevents_logger = logging.getLogger("fsevents")
# #     fsevents_logger.setLevel(logging.WARNING)

# #     main()


##################################################################


# # USAGE
# # To read and write back out to video:
# # python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4	 --output output/output_01.avi
# #
# # To read from webcam and write back out to disk:
# # python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel	--output output/webcam_output.avi

# # import the necessary packages
# from pyimagesearch.centroidtracker import CentroidTracker
# from pyimagesearch.trackableobject import TrackableObject
# from imutils.video import VideoStream
# from imutils.video import FPS
# import numpy as np
# import argparse
# import imutils
# import streamlit as st
# import time
# import dlib
# import cv2

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", default='mobilenet_ssd/MobileNetSSD_deploy.prototxt',
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", default='mobilenet_ssd/MobileNetSSD_deploy.caffemodel',
# 	help="path to Caffe pre-trained model")
# ap.add_argument("-i", "--input", type=str,default = 'videos/example_01.mp4',
# 	help="path to optional input video file")
# ap.add_argument("-o", "--output", type=str, default = 'output/output_01.avi',
# 	help="path to optional output video file")
# ap.add_argument("-c", "--confidence", type=float, default=0.4,
# 	help="minimum probability to filter weak detections")
# ap.add_argument("-s", "--skip-frames", type=int, default=30,
# 	help="# of skip frames between detections")
# args = vars(ap.parse_args())

# # MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
# MODEL_LOCAL_PATH = "./models/MobileNetSSD_deploy.caffemodel"
# # PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
# PROTOTXT_LOCAL_PATH ="./models/MobileNetSSD_deploy.prototxt.txt"

# # initialize the list of class labels MobileNet SSD was trained to
# # detect
# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
# 	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
# 	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
# 	"sofa", "train", "tvmonitor"]

# # load our serialized model from disk
# print("[INFO] loading model...")
# net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))

# # if a video path was not supplied, grab a reference to the webcam
# if not args.get("input", False):
# 	print("[INFO] starting video stream...")
# 	vs = VideoStream(src=0).start()
# 	time.sleep(2.0)

# # otherwise, grab a reference to the video file
# else:
# 	print("[INFO] opening video file...")
# 	vs = cv2.VideoCapture(args["input"])

# # initialize the video writer (we'll instantiate later if need be)
# writer = None

# # initialize the frame dimensions (we'll set them as soon as we read
# # the first frame from the video)
# W = None
# H = None

# # instantiate our centroid tracker, then initialize a list to store
# # each of our dlib correlation trackers, followed by a dictionary to
# # map each unique object ID to a TrackableObject
# ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
# trackers = []
# trackableObjects = {}

# # initialize the total number of frames processed thus far, along
# # with the total number of objects that have moved either up or down
# totalFrames = 0
# totalDown = 0
# totalUp = 0

# # start the frames per second throughput estimator
# fps = FPS().start()
# st.title("Real Time Object Tracking")
# st.subheader("Counting the no. of peron")
# frame_st = st.empty()
# # loop over frames from the video stream
# while True:
# 	# grab the next frame and handle if we are reading from either
# 	# VideoCapture or VideoStream
# 	frame = vs.read()
# 	frame = frame[1] if args.get("input", False) else frame

# 	# if we are viewing a video and we did not grab a frame then we
# 	# have reached the end of the video
# 	if args["input"] is not None and frame is None:
# 		break

    
# 	# resize the frame to have a maximum width of 500 pixels (the
# 	# less data we have, the faster we can process it), then convert
# 	# the frame from BGR to RGB for dlib
# 	frame = imutils.resize(frame, width=500)
# 	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 	# if the frame dimensions are empty, set them
# 	if W is None or H is None:
# 		(H, W) = frame.shape[:2]

# 	# if we are supposed to be writing a video to disk, initialize
# 	# the writer
# 	if args["output"] is not None and writer is None:
# 		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# 		writer = cv2.VideoWriter(args["output"], fourcc, 30,
# 			(W, H), True)

# 	# initialize the current status along with our list of bounding
# 	# box rectangles returned by either (1) our object detector or
# 	# (2) the correlation trackers
# 	status = "Waiting"
# 	rects = []

# 	# check to see if we should run a more computationally expensive
# 	# object detection method to aid our tracker
# 	if totalFrames % args["skip_frames"] == 0:
# 		# set the status and initialize our new set of object trackers
# 		status = "Detecting"
# 		trackers = []

# 		# convert the frame to a blob and pass the blob through the
# 		# network and obtain the detections
# 		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
# 		net.setInput(blob)
# 		detections = net.forward()

# 		# loop over the detections
# 		for i in np.arange(0, detections.shape[2]):
# 			# extract the confidence (i.e., probability) associated
# 			# with the prediction
# 			confidence = detections[0, 0, i, 2]

# 			# filter out weak detections by requiring a minimum
# 			# confidence
# 			if confidence > args["confidence"]:
# 				# extract the index of the class label from the
# 				# detections list
# 				idx = int(detections[0, 0, i, 1])

# 				# if the class label is not a person, ignore it
# 				if CLASSES[idx] != "person":
# 					continue

# 				# compute the (x, y)-coordinates of the bounding box
# 				# for the object
# 				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
# 				(startX, startY, endX, endY) = box.astype("int")

# 				# construct a dlib rectangle object from the bounding
# 				# box coordinates and then start the dlib correlation
# 				# tracker
# 				tracker = dlib.correlation_tracker()
# 				rect = dlib.rectangle(startX, startY, endX, endY)
# 				tracker.start_track(rgb, rect)

# 				# add the tracker to our list of trackers so we can
# 				# utilize it during skip frames
# 				trackers.append(tracker)

# 	# otherwise, we should utilize our object *trackers* rather than
# 	# object *detectors* to obtain a higher frame processing throughput
# 	else:
# 		# loop over the trackers
# 		for tracker in trackers:
# 			# set the status of our system to be 'tracking' rather
# 			# than 'waiting' or 'detecting'
# 			status = "Tracking"

# 			# update the tracker and grab the updated position
# 			tracker.update(rgb)
# 			pos = tracker.get_position()

# 			# unpack the position object
# 			startX = int(pos.left())
# 			startY = int(pos.top())
# 			endX = int(pos.right())
# 			endY = int(pos.bottom())

# 			# add the bounding box coordinates to the rectangles list
# 			rects.append((startX, startY, endX, endY))

# 	# draw a horizontal line in the center of the frame -- once an
# 	# object crosses this line we will determine whether they were
# 	# moving 'up' or 'down'
# 	cv2.line(frame, (0, 3*H // 4), (W, 3* H // 4), (0, 255, 255), 2)
# 	cv2.putText(frame, "GATE", (0,3* H // 4),
# 			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) ############################@##########

# 	# use the centroid tracker to associate the (1) old object
# 	# centroids with (2) the newly computed object centroids
# 	objects = ct.update(rects)

# 	# loop over the tracked objects
# 	for (objectID, centroid) in objects.items():
# 		# check to see if a trackable object exists for the current
# 		# object ID
# 		to = trackableObjects.get(objectID, None)

# 		# if there is no existing trackable object, create one
# 		if to is None:
# 			to = TrackableObject(objectID, centroid)

# 		# otherwise, there is a trackable object so we can utilize it
# 		# to determine direction
# 		else:
# 			# the difference between the y-coordinate of the *current*
# 			# centroid and the mean of *previous* centroids will tell
# 			# us in which direction the object is moving (negative for
# 			# 'up' and positive for 'down')
# 			y = [c[1] for c in to.centroids]
# 			direction = centroid[1] - np.mean(y)
# 			to.centroids.append(centroid)

# 			# check to see if the object has been counted or not
# 			if not to.counted:
# 				# if the direction is negative (indicating the object
# 				# is moving up) AND the centroid is above the center
# 				# line, count the object
# 				if direction < 0 and centroid[1] < H // 2:
# 					totalUp += 1
# 					to.counted = True

# 				# if the direction is positive (indicating the object
# 				# is moving down) AND the centroid is below the
# 				# center line, count the object
# 				elif direction > 0 and centroid[1] > H // 2:
# 					totalDown += 1
# 					to.counted = True

# 		# store the trackable object in our dictionary
# 		trackableObjects[objectID] = to

# 		# draw both the ID of the object and the centroid of the
# 		# object on the output frame
# 		text = "ID {}".format(objectID)
# 		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
# 			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# 		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

# 	# construct a tuple of information we will be displaying on the
# 	# frame
# 	info = [
# 		("OUT", totalUp),
# 		("IN", totalDown),
# 		("Status", status),
# 	]

# 	# loop over the info tuples and draw them on our frame
# 	for (i, (k, v)) in enumerate(info):
# 		text = "{}: {}".format(k, v)
# 		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
# 			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# 	# check to see if we should write the frame to disk
# 	if writer is not None:
# 		writer.write(frame)

# 	# show the output frame
# 	frame_st.image(frame)

#     # frame_st.image(frame)
# 	key = cv2.waitKey(1) & 0xFF
    

# 	# if the `q` key was pressed, break from the loop
# 	if key == ord("q"):
# 		break

# 	# increment the total number of frames processed thus far and
# 	# then update the FPS counter
# 	totalFrames += 1
# 	fps.update()

# # stop the timer and display FPS information
# fps.stop()
# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# # check to see if we need to release the video writer pointer
# if writer is not None:
# 	writer.release()

# # if we are not using a video file, stop the camera video stream
# if not args.get("input", False):
# 	vs.stop()

# # otherwise, release the video file pointer
# else:
# 	vs.release()

# # close any open windows
# cv2.destroyAllWindows()


################################################################################################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
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

def trackbar():
	h_min = st.slider('Hue min',0,179,0)
	h_max = st.slider('Hue max',0,179,179)
	s_min = st.slider('Sat min',0,255,0)
	s_max = st.slider('Sat max',0,255,255)
	v_min = st.slider('Val min',0,255,156)
	v_max = st.slider('Val max',0,255,255)
	return h_min,h_max,s_min,s_max,v_min,v_max 

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

# # def TrackBar():
# #     cv2.namedWindow('TrackBars')
# #     cv2.resizeWindow('TrackBars', 640, 240)
# #     cv2.createTrackbar('Hue Min','TrackBars',0,179,empty)
# #     cv2.createTrackbar('Hue Max','TrackBars',179,179,empty)
# #     cv2.createTrackbar('Sat Min','TrackBars',0,255,empty)
# #     cv2.createTrackbar('Sat Max','TrackBars',255,255,empty)
# #     cv2.createTrackbar('Val Min','TrackBars',156,255,empty)
# #     cv2.createTrackbar('Val Max','TrackBars',255,255,empty)
# # 	# confidence_threshold = st.slider(
# #     #     "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
# #     # )
# def HSV(img):
#     # TrackBar()
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	
#     h_min = st.slider('Hue Min',0,179,0)
#     h_max = st.slider('Hue Max',0,179,179)
#     s_min = st.slider('Sat Min',0,255,0)
#     s_max = st.slider('Sat Max',0,255,255)
#     v_min = st.slider('Val Min',0,255,156)
#     v_max = st.slider('Val Max',0,255,255)
    
#     lower = np.array([h_min, s_min, v_min])
#     upper = np.array([h_max, s_max, v_max])
#     mask = cv2.inRange(img_hsv, lower, upper)
#     img_result = cv2.bitwise_and(img,img,mask=mask)
#     return mask


# def getCountours(img_h, img_c):
#     contours, hierarchy = cv2.findContours(img_c, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         cv2.drawContours(img_contour1, cnt, -1, (0,255,0),3)
#         peri = cv2.arcLength(cnt,True)
#         approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
#         objCor = len(approx)
#         x, y, w, h = cv2.boundingRect(approx)
#         cv2.rectangle(img_detect1,(x,y),(x+w,y+h),(255,0,0),2)
    
#     contours, hierarchy = cv2.findContours(img_h, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         cv2.drawContours(img_contour2, cnt, -1, (0,255,0),3)
#         peri = cv2.arcLength(cnt,True)
#         approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
#         objCor = len(approx)
#         x, y, w, h = cv2.boundingRect(approx)
#         cv2.rectangle(img_detect2,(x,y),(x+w,y+h),(255,0,0),2)

# def putLabel(img, text):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(img, text, (10,400), font, 1,(255,255,255),2,cv2.LINE_AA)
        
# img = cv2.imread('pinwheel2.png')
# img_contour1 = img.copy()
# img_contour2 = img.copy()
# img_detect1 = img.copy()
# img_detect2 = img.copy()
# img_hsv = HSV(img)
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img_blur = cv2.GaussianBlur(img_gray,(7,7),1)
# img_canny = cv2.Canny(img,300,300)

# getCountours(img_hsv, img_canny)

# putLabel(img, 'Original')
# putLabel(img_gray, 'Gray Scaling')
# putLabel(img_blur, 'Blured Image')
# putLabel(img_canny, 'OpenCV mask')
# putLabel(img_hsv, 'Our mask')
# putLabel(img_contour1, 'OpenCV contour')
# putLabel(img_contour2, 'Our contour')
# putLabel(img_detect1,'OpenCV detected')
# putLabel(img_detect2,'We detected')

# img_ver = stackImages(0.60,([img,img_gray,img_blur],
#                             [img_canny, img_contour1,img_detect1],
#                             [img_hsv, img_contour2, img_detect2]))
# # cv2.imshow('Output',img_ver)
# st.image(img_ver)

# cv2.waitKey(0)
# cv2.destroyAllWindows()