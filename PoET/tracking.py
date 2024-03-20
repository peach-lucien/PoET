
import matplotlib.pyplot as plt
import numpy as np

import skvideo.io
import mediapipe as mp

import os

from tqdm import tqdm

import pkg_resources

from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

from PoET.mediapipe_landmarks import prepare_empty_dataframe



def load_models(min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,):
    
    HAND_MODEL = pkg_resources.resource_filename('PoET', 'models/hand_landmarker.task')
    
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # hand detection
    base_options = mp.tasks.BaseOptions(model_asset_path=HAND_MODEL)
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    options = HandLandmarkerOptions(base_options=base_options,
                                    num_hands=2,
                                    running_mode=VisionRunningMode.VIDEO)
    options.min_hand_detection_confidence = min_hand_detection_confidence
    options.min_tracking_confidence = min_tracking_confidence
    hands = HandLandmarker.create_from_options(options)
    
    
    POSE_MODEL = pkg_resources.resource_filename('PoET', 'models/pose_landmarker_full.task')
    
    # pose detection
    base_options = mp.tasks.BaseOptions(model_asset_path=POSE_MODEL)
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    options = PoseLandmarkerOptions(base_options=base_options,
                                    running_mode=VisionRunningMode.VIDEO,
                                    min_pose_detection_confidence=0.8)
    pose = PoseLandmarker.create_from_options(options)    
    
    return hands, pose
    

def track_video_list(video_list, 
                     output_folder='./tracking/', 
                     overwrite=False, 
                     verbose=True, 
                     make_csv=True, 
                     make_video=True, 
                     world_coords=False,
                     min_tracking_confidence=0.5,
                     min_hand_detection_confidence=0.5,
                     ):    
    """ video list but have full paths """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)    
    
    for i, video in enumerate(video_list):
                
        # reload models on each iteration (reset timestamp)
        hands, pose = load_models(min_hand_detection_confidence=min_hand_detection_confidence,
                                  min_tracking_confidence=min_tracking_confidence)
        
        # extract video name from full path
        video_name = os.path.basename(video).split('.')[0]    
    
        if verbose:
            print(video)    
        
        if not overwrite:
            if os.path.isfile(output_folder + video_name + '_MPtracked.csv'):
                if verbose:
                    print('CSV file already made - skipping this video.')
                continue

        track_video(video, pose, hands, output_folder=output_folder, make_csv=make_csv, make_video=make_video, world_coords=world_coords) 
        
    return



def track_video(video, pose, hands, output_folder='./', make_csv=True, make_video=False, plot=False, world_coords=True):
    """ track a single video """    
    
    # print name
    print(video)
    
    # extract video name from full path
    video_name = os.path.basename(video).split('.')[0]       

    # defining a video reader
    videogen = list(skvideo.io.vreader(video))
    
    # defining a video writer
    writer = skvideo.io.FFmpegWriter(output_folder + video_name + '_MPtracked.mp4')
    
    # extract video meta data if you want it
    metadata = skvideo.io.ffprobe(video)    
    
    # extract framerate of video
    fs = int(metadata['video']['@r_frame_rate'][:2])
    
    # construct empty dataframe for filling with tracking results
    marker_df, marker_mapping = prepare_empty_dataframe(hands='both',pose=True)
    
    # looping over the frames in the video
    for i, image in enumerate(tqdm(videogen, total=len(videogen))):        
        
        # converted for MP format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # get frame time in ms
        frame_ms = fs*i
        
        # running mediapipe predictions
        #results_face_mesh = face_mesh.detect_for_video(mp_image,frame_ms)
        results_hands = hands.detect_for_video(mp_image, frame_ms)
        results_pose = pose.detect_for_video(mp_image, frame_ms)
        #results_pose = pose.process(image) 
                
        # creating a copy of image
        annotated_image = image.copy()      
        
        if world_coords:
            img_h, img_w = (1,1) # the world coordinates are supposedly in metres
        else:
            img_h, img_w = image.shape[:2] # the non-world coordinates should be converted into pixels

                
        # if the tracking was successful
        if results_pose.pose_world_landmarks:        
            # loop over faces detected (hopefully just one)        
            annotated_image = draw_pose_landmarks_on_image(annotated_image, results_pose)
            
            if world_coords:
                out = results_pose.pose_world_landmarks[0]
            else:
                out = results_pose.pose_landmarks[0]
            
            # only looping over the first pose (assuming there is only one person in the frame)
            for l, landmark in enumerate(out):
                
                marker = marker_mapping['pose'][l]
                marker_df.loc[i,(marker,'x')] = landmark.x*img_w
                marker_df.loc[i,(marker,'y')] = landmark.y*img_h
                marker_df.loc[i,(marker,'z')] = landmark.z
                marker_df.loc[i,(marker,'visibility')] = landmark.visibility
                marker_df.loc[i,(marker,'presence')] = landmark.presence
                   
            
        if results_hands.hand_landmarks:
            annotated_image = draw_hand_landmarks_on_image(annotated_image, results_hands)
            
            if world_coords:
                out = results_hands.hand_world_landmarks
            else:
                out = results_hands.hand_landmarks
            
            for h, hand in enumerate(out):
                
                handedness = results_hands.handedness[h][0].display_name
                
                # # due to some weird bug, its the wrong way around
                # if handedness=='Right':
                #     handedness='Left'
                # elif handedness=='Left':
                #     handedness='Right'
                
                for l, landmark in enumerate(hand):
                    
                    marker = marker_mapping[handedness + '_hand'][l]                        
                    marker_df.loc[i,(marker,'x')] = landmark.x*img_w
                    marker_df.loc[i,(marker,'y')] = landmark.y*img_h
                    marker_df.loc[i,(marker,'z')] = landmark.z
                    marker_df.loc[i,(marker,'visibility')] = landmark.visibility
                    marker_df.loc[i,(marker,'presence')] = landmark.presence

    
    
        # only plotting if set to True
        if plot: 
            plt.figure()
            #plt.imshow(np.flip(annotated_image, axis=1))   
            plt.imshow(annotated_image) 
            
        # write annotated image to video
        if make_video:
            writer.writeFrame(annotated_image)
            
    if make_csv:
        marker_df.to_csv(output_folder + video_name + '_MPtracked.csv')

    # closing the writer 
    writer.close()
    
    return


def draw_pose_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        
        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          pose_landmarks_proto,
          solutions.pose.POSE_CONNECTIONS,
          solutions.drawing_styles.get_default_pose_landmarks_style())
        
    return annotated_image


def draw_hand_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    
    # Loop through the detected poses to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        
        # Draw the pose landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style())
        
    return annotated_image


def draw_face_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    
    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
          
        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
          
        solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
        
        solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
        
        solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                  landmark_drawing_spec=None,
                  connection_drawing_spec=mp.solutions.drawing_styles
                  .get_default_face_mesh_iris_connections_style())
    
    return annotated_image
