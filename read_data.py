from typing import Literal
import pandas

from datetime import date as dt

def read_measurement(type:Literal["ina", "bobina"]="ina", number:int=42, date=None):
    if date is None: date = dt.today()
    folder_path = f"~/Documents/measurements_gpu/measurements_{type}/{date}/"
    data = pandas.read_csv(folder_path + f"{number:03d}.csv")
    return data
