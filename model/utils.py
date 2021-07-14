from . import SNN_cnn, SNN_cnn_6Dof


model_map = {
    'SNN_cnn': {
        'net': SNN_cnn.SNN_Model,
    },
    'SNN_cnn_6Dof': {
        'net': SNN_cnn_6Dof.SNN_Model,
    },
}


def getNetwork(model_type: str, simulation_params):
    return model_map[model_type]['net'](simulation_params)
