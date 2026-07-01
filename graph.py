import os

import matplotlib.pyplot as pyplot
import pandas
import numpy

pyplot.rcParams['figure.figsize'] = [19.2, 10.8]
pyplot.rcParams['savefig.dpi'] = 100

from pathlib import Path
import os

def plot_graph(
        values,
        title:str,
        show:bool=False,
        path_save:str|None=None,
        figure_name:str|None=None,
        legend=True
    ):
    fig, ax1 = pyplot.subplots(layout='constrained')
    ax1.set_facecolor((0.95, 0.9, 0.85, 1))
    ax2 = ax1.twinx()

    #ax1.axvspan(min(times), max(times), facecolor=(0.95, 0.9, 0.85, 1), alpha=0.5)
    color_w = "green"
    color_j = "blue"
    for value in values:
        times = value["time"]
        measurements = value["measurement"]
        # plot the power consumed line
        ax1.plot(times, measurements, label="Power (W)", linestyle="solid", linewidth=1, color=color_w, alpha=1.0)
        ax1.scatter(times, measurements, s=3, c=color_w, alpha=1.0)
        

        # plot the horizontal dotted lines
        #measurement_mean = numpy.mean(measurements)
        measurement_min = min(measurements)
        measurement_max = max(measurements)

        #ax1.axhline(y=measurement_mean, label=f"Média = {measurement_mean:.3f} W", linestyle="--", color="orange", alpha=1.0)
        #ax1.axhline(y=measurement_min,  label=  f"Min = { measurement_min:.3f} W", linestyle=":",  color="red"   , alpha=1.0)
        #ax1.axhline(y=measurement_max,  label=  f"Max = { measurement_max:.3f} W", linestyle=":",  color="red"   , alpha=1.0)

        # calculates the total energy consumed through simple integration
        power = [0]
        i = 1
        measurement_count = len(measurements)
        while i < measurement_count:
            # calculates the area of the trapezium from one point to the other
            power.append(float(
                power[-1] +
                (measurements[i] + measurements[i-1]) * (times[i] - times[i-1]) / 2
            ))
            i += 1
        print("Total power:", power[-1], "W")

        # Configure labels and axis of total consumed plot
        ax2.plot(times, power, label=f"Energia Acumulada ({(power[-1]):.2f} J)", linestyle="solid", linewidth=1.5, color=color_j, alpha=1.0)
        color_w = "red"
        color_j = "orange"

    # Configure labels and axis of power plot
    ax1.set_xlabel("Tempo (s)", fontsize=40)
    ax1.set_ylabel("Power (W)", fontsize=40, color=color_w)
    ax1.tick_params("y", colors=color_w)
    ax1.tick_params("both", labelsize=40)
    ax1.grid(True)

    min_timestamp = min(times)
    max_timestamp = max(times)
    
    #padding = (
    #    (max_timestamp - min_timestamp) * 0.025,
    #    (measurement_max - measurement_min) * 0.05,
    #)
    
    #ax1.axis([
    #    min_timestamp - padding[0],
    #    max_timestamp + padding[0],
    #    measurement_min - padding[1],
    #    measurement_max + padding[1]
    #]) 

    ax2.set_facecolor("none")
    ax2.set_ylabel(f"Energia Acumulada (J)", color=color_j)
    ax2.tick_params("y", colors=color_j)
    print("Total power:", power[-1], "W")


    # joins both plots together
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1   + labels2

    if (legend): ax1.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=40)
    ax1.set_title(title, fontsize=50)
    
    if path_save:
        os.makedirs(path_save, exist_ok=True)
        if path_save[-1] != "/": path_save += "/"
        pyplot.savefig(path_save + figure_name+".png", format="png")
    
    if show:    pyplot.show()
    pyplot.close("all")
