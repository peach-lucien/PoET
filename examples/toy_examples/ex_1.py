
import pandas as pd

from PoET import POET

# =============================================================================
# Perform tracking of videos
# =============================================================================

# instantiate POET object with path to video folder
folder = './data/'
poet = POET(folder)

# load metadata
metadata = pd.read_excel('./data/scaling_parameters.xlsx',header=0, index_col=0)
metadata['scaling_factor'] = metadata.iloc[:,2]/metadata.iloc[:,1]

# run tracking and tremor analysis for each video in the list
features = poet.run(sampling_frequency = metadata.framerate,
                    scaling_factor = metadata.scaling_factor,
                    tremor_type='postural')












