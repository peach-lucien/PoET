
import os
import glob

from PoET.tracking import track_video_list
from PoET.kinematics import extract_tremor
from PoET.preprocessing import construct_data
from PoET.features import extract_tremor_features, assign_hand_time_periods


class POET:
    def __init__(self, video_folder=None):
        self.video_folder = video_folder
        
        if video_folder:
            self.video_list = self.identify_videos()
        else:
            self.video_list = None
        
        self.output_folder='./tracking/'
        self.patient_collection = None

    def identify_videos(self):
        """Identify all video files in the given folder."""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv','.MOV')
        return [os.path.join(self.video_folder, f) for f in os.listdir(self.video_folder)
                if f.endswith(video_extensions)]
    
    def run(self, 
            output_folder='./tracking/',
            make_video=True,
            world_coords=False,
            min_tracking_confidence=0.5,
            min_hand_detection_confidence=0.5,
            csv_files=None,
            sampling_frequency=25,
            scaling_factor=1,
            labels=None,
            save=True,
            patient_collection=None,
            tremor_type='postural',
            ):
        """ End-to-end running of the code """
        
        # run tracking
        self.run_tracking(output_folder=output_folder,
                        make_video=make_video,
                        world_coords=world_coords,
                        min_tracking_confidence=min_tracking_confidence,
                        min_hand_detection_confidence=min_hand_detection_confidence)
        
        # loading markers
        self.load_tracking(csv_files=csv_files,
                            sampling_frequency=sampling_frequency,
                            scaling_factor=scaling_factor,
                            labels=labels,)
        
        # run analysis
        features = self.analyse_tremors(save=save,
                            patient_collection= patient_collection,
                            tremor_type=tremor_type)
        
        return features

    def run_tracking(self,
                     output_folder='./tracking/',
                     make_video=True,
                     world_coords=False,
                     min_tracking_confidence=0.5,
                     min_hand_detection_confidence=0.5):
        
        # set the folder for storing the tracked markers
        self.output_folder=output_folder
        
        # run tracking for each video in the list
        track_video_list(self.video_list, 
                         output_folder=output_folder,
                         make_video=make_video, 
                         world_coords=world_coords,
                         min_tracking_confidence=min_tracking_confidence,
                         min_hand_detection_confidence=min_hand_detection_confidence,
                         )
        return
    
    def load_tracking(self,                         
                      csv_files=None,
                      sampling_frequency=25,
                      scaling_factor=1,
                      labels=None,):
        
        # finding all tracked csv files
        if csv_files is None:
            csv_files = glob.glob(self.output_folder + '*.csv', recursive=True)
        elif isinstance(csv_files, str):
            csv_files = glob.glob(csv_files + '*.csv', recursive=True)

        # assigning a sampling frequency for each file
        if isinstance(sampling_frequency,int):
            sampling_frequency = [sampling_frequency] * len(csv_files)
        elif len(sampling_frequency) != len(csv_files):
            sampling_frequency = get_metadata(sampling_frequency, csv_files)

        # assigning a scaling factor for each csv file
        if isinstance(scaling_factor,int):
            scaling_factor = [scaling_factor] * len(csv_files)
        elif len(scaling_factor) != len(csv_files):
            scaling_factor = get_metadata(scaling_factor, csv_files)

        # assigning a None label to each csv file
        if labels is None:
            labels = [None] * len(csv_files)
            
        # construct dataset from csv files
        pc = construct_data(csv_files,
                            sampling_frequency,
                            labels=labels,
                            scaling_factor=scaling_factor)
        
        self.patient_collection = pc
        
        return

    def analyse_tremors(self, 
                        save=True,
                        patient_collection=None,
                        tremor_type='postural'):
        """Main function call that processes the videos by calling the relevant methods."""

        if patient_collection is not None:
            self.patient_collection = patient_collection
                        
        # extract tremor 
        self.patient_collection = extract_tremor(self.patient_collection)
        
        # try to automatically define which hand is active
        self.patient_collection = assign_hand_time_periods(self.patient_collection)  
        
        # extract features        
        features = extract_tremor_features(self.patient_collection, tremor_type=tremor_type)
        
        if save:
            features.to_csv('./features.csv')
        
        return features
    
    
def get_metadata(df, file_paths):
    matched_values = []
    
    for file_path in file_paths:
        # Extract the filename from the path
        filename = os.path.basename(file_path)
        
        # Check if any index (substring) matches the filename
        for index in df.index:
            if index in filename:
                # If a match is found, extract the value from 'col1' and append to the list
                matched_values.append(df[index])
                break  # Assumes only one match per filename, remove if multiple matches are possible
    return matched_values