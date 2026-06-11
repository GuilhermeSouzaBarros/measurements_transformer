param_grids_models = {
    "LinReg": {
        'tol' : [1e-10, 1e-6]
    },
    "SVR": {
        'C': [2**i for i in range(0, 2)]
    }
}

def param_grid_retrieve(name_model:str):
    return param_grids_models[name_model]
