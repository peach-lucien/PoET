import pandas as pd
from itertools import product

hand_landmarks =  {0:'wrist',
                   1:'thumb_cmc',
                   2:'thumb_mcp',
                   3:'thumb_ip',
                   4:'thumb_tip',
                   5:'index_finger_mcp',
                   6:'index_finger_pip',
                   7:'index_finger_dip',
                   8:'index_finger_tip',
                   9:'middle_finger_mcp',
                   10:'middle_finger_pip',
                   11:'middle_finger_dip',
                   12:'middle_finger_tip',
                   13:'ring_finger_mcp',
                   14:'ring_finger_pip',
                   15:'ring_finger_dip',
                   16:'ring_finger_tip',
                   17:'pinky_mcp',
                   18:'pinky_pip',
                   19:'pinky_dip',
                   20:'pinky_tip',
                   }

pose_landmarks =   {0: 'nose',
                    1: 'left_eye_inner',
                    2: 'left_eye',
                    3: 'left_eye_outer',
                    4: 'right_eye_inner',
                    5: 'right_eye',
                    6: 'right_eye_outer',
                    7: 'left_ear',
                    8: 'right_ear',
                    9: 'mouth_left',
                    10:' mouth_right',
                    11: 'left_shoulder',
                    12: 'right_shoulder',
                    13: 'left_elbow',
                    14: 'right_elbow',
                    15: 'left_wrist',
                    16: 'right_wrist',
                    17: 'left_pinky',
                    18: 'right_pinky',
                    19: 'left_index',
                    20: 'right_index',
                    21: 'left_thumb',
                    22: 'right_thumb',
                    23: 'left_hip',
                    24: 'right_hip',
                    25: 'left_knee',
                    26: 'right_knee',
                    27: 'left_ankle',
                    28: 'right_ankle',
                    29: 'left_heel',
                    30: 'right_heel',
                    31: 'left_foot_index',
                    32: 'right_foot_index',
                    }

def prepare_empty_dataframe(hands='both',pose=True,face_mesh=False):
    """ creates empty dataframe for tracking """
    
    marker_dictionaries = {}
    
    marker_dictionaries['Left_hand'] = {u: hand_landmarks[u] + '_left' for u in hand_landmarks}
    marker_dictionaries['Right_hand'] = {u: hand_landmarks[u] + '_right' for u in hand_landmarks}
    marker_dictionaries['pose'] = pose_landmarks

    columns = []
    if hands:
        if hands == 'both':            
            columns += [hand_landmarks[u]+'_left' for u in hand_landmarks] + [hand_landmarks[u]+'_right' for u in hand_landmarks]
        elif hands == 'left':
            columns += [hand_landmarks[u]+'_left' for u in hand_landmarks]
        elif hands == 'right':
            columns += [hand_landmarks[u]+'_right' for u in hand_landmarks]
            
    if pose:
        columns += [pose_landmarks[u] for u in pose_landmarks]
    
    multi_columns = list(product(columns,['x','y','z','visibility','presence']))
        
    dataframe = pd.DataFrame(columns=multi_columns)        
    dataframe.columns = pd.MultiIndex.from_tuples(dataframe.columns, names=['marker','subindex'])
    
    return dataframe, marker_dictionaries