"""input/output functions."""
import os
import pickle
from pathlib import Path

def save_object(obj, folder=".", filename="./obj"):
    """Save general object in a pickle."""
    if not Path(filename).parent.is_dir():
        Path(filename).parent.mkdir()

    pickle.dump(
        obj,
        open(os.path.join(folder, filename + ".pkl"), "wb"),
    )

def load_object(folder=".", filename="obj"):
    """Save the features in a pickle."""
    return pickle.load(open(os.path.join(folder, filename + ".pkl"), "rb"))


def save_fitted_model(fitted_model, scaler, feature_names, folder=".", filename="./model"):
    """Save the features in a pickle."""
    if not Path(filename).parent.is_dir():
        Path(filename).parent.mkdir()

    pickle.dump(
        [fitted_model, scaler, feature_names],
        open(os.path.join(folder, filename + ".pkl"), "wb"),
    )


def load_fitted_model(folder=".", filename="model"):
    """Save the features in a pickle."""
    return pickle.load(open(os.path.join(folder, filename + ".pkl"), "rb"))


def save_unsupervised_behavioural_model(models, folder="./outputs/unsupervised_behavioural_model/"):
    """Save the unsupervised behavioural model in a pickle."""
    if not Path(folder).is_dir():
        Path(folder).mkdir()

    models['autoencoder'].save(folder+'autoencoder')
    models['encoder'].save(folder+'encoder')    

    pickle.dump(
        models['model_parameters'],
        open(os.path.join(folder, "model_parameters.pkl"), "wb"),
    )
    pickle.dump(
        models['cluster_model'],
        open(os.path.join(folder, "cluster_model.pkl"), "wb"),
    )    



def save_fitted_behavioural_model(models, folder="./outputs/behavioural_model/"):
    """Save the behavioural model  in a pickle."""
    if not Path(folder).is_dir():
        Path(folder).mkdir()

    if not Path(folder+'xgboost_model/').is_dir():
        Path(folder+'xgboost_model').mkdir()

    if 'xgboost' in models['fitted_models'].keys():
        pickle.dump(
            models['fitted_models']['xgboost'],
            open(os.path.join(folder, "xgboost_model.pkl"), "wb"),
        )
    
    if 'lstm' in models['fitted_models'].keys():
        models['fitted_models']['lstm'].save(folder+'lstm_model')
        
    if 'convlstm' in models['fitted_models'].keys():    
        models['fitted_models']['convlstm'].save(folder+'conv_lstm_model')


    pickle.dump(
        models['model_parameters'],
        open(os.path.join(folder, "model_parameters.pkl"), "wb"),
    )


# def load_fitted_behavioural_model(folder="./outputs/behavioural_model/"):
#     """ load the behavioural model in a pickle."""
    
#     models = {
#             'fitted_models':{},
#             'model_parameters': None,
#              }
    
#     models['model_parameters'] = pickle.load(open(os.path.join(folder,"model_parameters.pkl"), "rb"))

#     if models['model_parameters']['xgboost']:
#         models['fitted_models']['xgboost'] = pickle.load(open(os.path.join(folder,"xgboost_model.pkl"), "rb"))

#     if models['model_parameters']['lstm']:        
#         models['fitted_models']['lstm'] = tf.keras.models.load_model(folder+'lstm_model')

#     if models['model_parameters']['conv_lstm']:
#         models['fitted_models']['convlstm'] = tf.keras.models.load_model(folder+'conv_lstm_model')
   
#     return models







