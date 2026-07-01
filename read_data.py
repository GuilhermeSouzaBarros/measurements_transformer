from typing import Literal
import pandas
from tqdm import tqdm
from datetime import date as dt

def read_measurement(type:Literal["ina", "bobina"]="ina", number:int=42):
    folder_path = f"measurements/{type}/"
    data = pandas.read_csv(folder_path + f"{number:03d}.csv")
    return data

def read_measurements(type:Literal["ina", "bobina"]="ina", number_min:int=1, number_max:int=760):
    folder_path = f"measurements/{type}/"
    datas = []
    for i in tqdm(range(number_min, number_max+1), "Loading data", ascii=True):
        data = pandas.read_csv(folder_path + f"{i:03d}.csv")
        datas.append(data)

    return datas
