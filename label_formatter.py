from tqdm import tqdm
import numpy

def format_label(measurement_real):
    data = []
    for real in measurement_real:
        label = {"time": real["time"], "label": numpy.mean(real["measurement"])}
        data.append(label)

    return data
