
import pandas as pd

from PoET import POET

# =============================================================================
# Perform tracking of videos
# =============================================================================

# instantiate POET object with path to video folder
folder = './data/'
poet = POET(folder)

# run tracking for each video in the list
poet.run_tracking()


# =============================================================================
# Prepare list of csv files and their conditions
# =============================================================================

# load metadata
metadata = pd.read_excel('./data/scaling_parameters.xlsx',header=0, index_col=0)
metadata['scaling_factor'] = metadata.iloc[:,2]/metadata.iloc[:,1]

# load markers and track tremors
poet.load_tracking(sampling_frequency = metadata.framerate,
                   scaling_factor = metadata.scaling_factor,
                   )

# analyse tremors
features = poet.analyse_tremors(tremor_type='postural')










