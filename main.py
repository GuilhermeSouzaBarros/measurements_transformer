import pandas as pd
import numpy as np
from tqdm import tqdm
from label_formatter import format_label

import read_data
import transformers
from args_handler import get_args
from graph import plot_graph

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

if __name__ == "__main__":
    name, model, param_grid = get_args(["w", "LinReg"])
    
    print("Loading data...")
    
    data_ina = read_data.read_measurements("ina",    number_min=1, number_max=180)
    data_bob = read_data.read_measurements("bobina", number_min=1, number_max=180)
    # still need to fix when the measurements so both have the sime time window
    # And make sure it works when calculatting the offset between the timestamps 
    
    print("Finished Loading data...")

    print("Processing data")
    uniformer = transformers.UniformTimeInterval()
    spliter   = transformers.DataSpliter(16.75)
    am_extractor = transformers.AmExtractor()
    new_model = transformers.WraperRegression(model)

    data_ina = spliter.transform(uniformer.transform(data_ina, measurement="power (W)"))
    labels = []
    for i in tqdm(data_ina, "Processing labels", ascii=True): 
        labels.append(format_label(i))
    
    pipeline = make_pipeline(uniformer, spliter, am_extractor, new_model)
    pipeline.fit_transform(data_bob, labels)

    test_data = read_data.read_measurements("bobina", number_min=680, number_max=681)
    real_data = read_data.read_measurements("ina", number_min=680, number_max=681)
    real_data[0]["time"] = (real_data[0]["timestamp"] - real_data[0]["timestamp"][0]) / 1000000
    real_data[1]["time"] = (real_data[1]["timestamp"] - real_data[1]["timestamp"][0]) / 1000000
    real_data[0]["measurement"] = real_data[0]["power (W)"]
    real_data[1]["measurement"] = real_data[1]["power (W)"]
    prediction = pipeline.transform(test_data)
    for data, real in zip(prediction, real_data):
        plot_graph([data, real], "test", True, legend=False)
