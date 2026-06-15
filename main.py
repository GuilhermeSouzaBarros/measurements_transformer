import read_data
import transformers
from args_handler import get_args
from graph import plot_graph

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import numpy as np

if __name__ == "__main__":
    print("Loading data...")
    date = "2026-06-10"
    data_ina_low = read_data.read_measurements("ina",    number_min=1, number_max=300, date=date)
    data_bob_low = read_data.read_measurements("bobina", number_min=1, number_max=300, date=date)
    
    data_ina_high = read_data.read_measurements("ina",    number_min=350, number_max=760, date=date)
    data_bob_high = read_data.read_measurements("bobina", number_min=350, number_max=760, date=date)
    
    data_ina = data_ina_low + data_ina_high
    data_bob = data_bob_low + data_bob_high
    print("Finished Loading data...")

    print("Uniform data")
    uniform_data_tension = transformers.UniformTimeInterval()
    uniform_data_power = transformers.UniformTimeInterval("power (W)")
    
    data_bob_uniforme = uniform_data_tension.transform(data_bob)
    data_ina_uniforme = uniform_data_power.transform(data_ina)

    print("Split data")
    spliter = transformers.DataSpliter(4)
    data_bob_splitted = spliter.transform(data_bob_uniforme)
    data_ina_splitted = spliter.transform(data_ina_uniforme)

    print("Extracting AM data")
    am_extractor = transformers.AmExtractor()
    am_features = am_extractor.transform(data_bob_splitted)

    scaler = StandardScaler()

    print("Generating Label")
    label_power = []
    am_features_new = []
    for x, y in zip(am_features, data_ina_splitted):
        max_i = min(len(x), len(y)) - 1
        am_features_new.extend(x[:max_i])
        for sub_y in y[:max_i]:
            label_power.append(np.mean(sub_y["measurement"]))


    am_features_norm = scaler.fit_transform(am_features_new)
    #pw_extractor = transformers.PWExtractor()
    #pw_extractor.transform(data_bob)
    
    name, model, param_grid = get_args(["w", "LinReg"])
    grid_search = GridSearchCV(
        model, param_grid, scoring="r2", cv=5, refit=True,
        return_train_score=True, n_jobs=-1, verbose=2
    )
    
    print("Size of training:", len(am_features_norm))
    grid_search.fit(am_features_norm, label_power)

    print("Result:", grid_search.best_score_)
    print("Parameters:", grid_search.best_params_)
    model = grid_search.best_estimator_

    new_data_bob = [read_data.read_measurement("bobina", 324, date="2026-06-10")]

    uniform_data_new = uniform_data_tension.transform(new_data_bob)
    data_bob_splitted_new = spliter.transform(uniform_data_new)
    am_features_new = am_extractor.transform(data_bob_splitted_new)

    am_features_new_norm = scaler.transform(am_features_new[0])

    predicted = model.predict(am_features_new_norm)

    times = []
    measurements = []
    for data, label in zip(data_bob_splitted_new[0], predicted):
        times.extend(data["time"])
        measurements.extend([label]*len(data["time"]))

    plot_graph(
        times,
        measurements,
        title="Model regression predicition",
        show=True,
        legend=False
    )


