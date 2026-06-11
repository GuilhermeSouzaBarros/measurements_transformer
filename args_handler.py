from hyperparameters import param_grid_retrieve

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def get_args(argv):
    if len(argv) < 2:
        print("Input should be python3 main.py <Model>")
        exit()
        
    model_name = argv[1]
    if model_name == "LinReg":
        model = LinearRegression()
    elif model_name == "SVR":
        model = SVR()
    else:
        print("Available models: LinReg and SVR")
        exit()

    param_grid = param_grid_retrieve(model_name)

    name = model_name
    return name, model, param_grid
