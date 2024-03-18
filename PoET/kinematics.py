import pandas as pd

from PoET.utils import check_hand_


def extract_tremor(pc):
    
    # define set of distance for analysing tremor
    marker_features ={'right':[['index_finger_tip_right','x'], ['index_finger_tip_right','y'], # kinematic
                               ['middle_finger_tip_right','x'], ['middle_finger_tip_right','y'], # postural
                               ['right_elbow','x'], ['right_elbow','y']], # proximal
                        'left':[['index_finger_tip_left','x'], ['index_finger_tip_left','y'], # kinematic
                                ['middle_finger_tip_left','x'], ['middle_finger_tip_left','y'], # postural
                                ['left_elbow','x'], ['left_elbow','y']], # proximal
                        }    

    for patient in pc.patients:
        
        # identify tracked hand
        hands = check_hand_(patient)    
        
        # load their data
        pose_estimation = patient.pose_estimation
        
        # define structural features to compute
        structural_features = pd.DataFrame(index=pose_estimation.index)

        # loop over the chosen hands
        for hand in hands:
            
            features = marker_features[hand]            

            for f in features:
                structural_features.loc[:,'marker_'+'_'.join(f)] = pose_estimation[(f[0],f[1])]
        
        # store kinematics 
        patient.structural_features = structural_features    
    
    return pc
   


def extract_kinematic_tremor(pc):
    
    # define set of distance for analysing tapping
    marker_features ={'right':[['index_finger_tip_right','x'], ['index_finger_tip_right','y']],
                        'left':[['index_finger_tip_left','x'], ['index_finger_tip_left','y']],
                        }    

    for patient in pc.patients:
        
        # identify tracked hand
        hands = check_hand_(patient)    
        
        # load their data
        pose_estimation = patient.pose_estimation
        
        # define structural features to compute
        structural_features = pd.DataFrame(index=pose_estimation.index)

        # loop over the chosen hands
        for hand in hands:
            
            features = marker_features[hand]            

            for f in features:
                structural_features.loc[:,'marker_'+'_'.join(f)] = pose_estimation[(f[0],f[1])]
        
        # store kinematics 
        patient.structural_features = structural_features    
    
    return pc




def extract_postural_tremor(pc):
    
    # define set of distance for analysing tapping
    marker_features ={'right':[['middle_finger_tip_right','x'], ['middle_finger_tip_right','y']],
                        'left':[['middle_finger_tip_left','x'], ['middle_finger_tip_left','y']],
                        }    

    for patient in pc.patients:
        
        # identify tracked hand
        hands = check_hand_(patient)    
        
        # load their data
        pose_estimation = patient.pose_estimation
        
        # define structural features to compute
        structural_features = pd.DataFrame(index=pose_estimation.index)

        # loop over the chosen hands
        for hand in hands:
            
            features = marker_features[hand]            

            for f in features:
                structural_features.loc[:,'marker_'+'_'.join(f)] = pose_estimation[(f[0],f[1])]
        
        # store kinematics 
        patient.structural_features = structural_features    

    return pc

def extract_proximal_tremor(pc):
    
    # define set of distance for analysing tapping
    marker_features ={'right':[['right_elbow','x'], ['right_elbow','y']],
                        'left':[['left_elbow','x'], ['left_elbow','y']],
                        }    

    for patient in pc.patients:
        
        # identify tracked hand
        hands = check_hand_(patient)    
        
        # load their data
        pose_estimation = patient.pose_estimation
        
        # define structural features to compute
        structural_features = pd.DataFrame(index=pose_estimation.index)

        # loop over the chosen hands
        for hand in hands:
            
            features = marker_features[hand]            

            for f in features:
                structural_features.loc[:,'marker_'+'_'.join(f)] = pose_estimation[(f[0],f[1])]
        
        # store kinematics 
        patient.structural_features = structural_features    

    return pc


