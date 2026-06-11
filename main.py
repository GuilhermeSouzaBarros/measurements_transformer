import read_data
import transformers
from args_handler import get_args
from graph import plot_graph

from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    date = "2026-06-10"
    data_ina = read_data.read_measurement("ina", date=date)
    data_bob = read_data.read_measurement("bobina", date=date)

    spliter = transformers.DataSpliter(data_ina.shape[0])
    window = 10
    aligned_bob = spliter.transform(data_bob, data_ina)
    target_ina = data_ina["power (W)"].to_numpy()[window:-window]

    name, model, param_grid = get_args(["w", "LinReg"])
    grid_search = GridSearchCV(
        model, param_grid, scoring="r2", cv=5, refit=True,
        return_train_score=True, n_jobs=-1, verbose=2
    )
    grid_search.fit(aligned_bob, target_ina)

    print("Result:", grid_search.best_score_)
    print("Parameters:", grid_search.best_params_)
    model = grid_search.best_estimator_

    new_data_bob = read_data.read_measurement("bobina", 42)
    aligned_new = spliter.transform(new_data_bob,)
    predicted = model.predict(aligned_new)

    plot_graph(
        [250.192615 * i for i in range(window, len(predicted) + window)],
        predicted,
        title="Model regression predicition",
        show=True
    )
