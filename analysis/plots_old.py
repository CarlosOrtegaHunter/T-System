import random

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from analysis.formatters import BigNumberFormatter
from common.config import logger
from core import ContinuousSignal
from core import Event
from core import Equity

def standard_plot_preamble(start_date, end_date):
    # Format the x-axis to display dates correctly
    plt.xlabel("Date")
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjust date ticks
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format dates as Year-Month-Day
    plt.xticks(rotation=45, ha="right")

    plt.gca().yaxis.set_visible(False)  # Hide left (signal) y-axis

    # Add guiding lines to separate days
    dates= pd.date_range(start=start_date, end=end_date)
    separator_alpha = 0.2 if len(dates)<365 else 0.2 * 365/len(dates)
    for date in dates:
        if date.weekday()  == 6 or date.weekday() == 4:  # (Start of) Monday or Friday
            plt.axvline(x=(date + timedelta(hours=12)), color='gray', alpha=separator_alpha*2, linewidth=0.741)
        else:
            plt.axvline(x=(date + timedelta(hours=12)), color='gray', alpha=separator_alpha, linewidth=0.5)

def standard_signal_plot_setup(formatter : BigNumberFormatter, start_date : datetime, end_date : datetime, y_ticks=10):
    standard_plot_preamble(start_date, end_date)
    plt.gca().yaxis.set_visible(True)  # Show left (signal) y-axis
    # Set more ticks on the y-axis by adjusting MaxNLocator and apply format
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=y_ticks))
    plt.gca().yaxis.set_major_formatter(formatter.get_formatter())

# Function to plot the CHX volume for a given date range
def plot_volume(stock: Equity, start_date, end_date=None, min_volume=0, exchange="NYSE", cumulative=False) -> int:

    start_date, end_date = stock.time_window(start_date, end_date)
    data = stock.get_historical_volumes(start_date, end_date)

    # Set values to zero for days when volume is smaller than min_volume
    data.loc[data[exchange] < min_volume, exchange] = 0
    # TODO: pass filter into function instead of min volume?

    data = data.sort_values(by='date')
    if cumulative:
        data[exchange] = data[exchange].cumsum()

    # plt.figure(figsize=(16, 8))  # Increase figure size
    plt.bar(data['date'], data[exchange], width=1.0)

    # Format the x-axis to display dates correctly
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjust date ticks
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format dates as Year-Month-Day
    plt.xticks(rotation=45, ha="right")

    y_ticks = 10
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=y_ticks))

    formatter = BigNumberFormatter()
    plt.gca().yaxis.set_major_formatter(formatter.get_formatter())

    # Add title and labels
    title = f"{exchange} Volume Per Day ({start_date} to {end_date})"
    if cumulative:
        title += " - Cumulative"
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(f"{exchange} Volume (in millions)")
    return data[exchange].values[-1]

#TODO: take Equity and ContinualSignal objects as arguments, not a dataframe!
# alternatively take a label to get the signal from the Equity object
# signal['date'] = pd.to_datetime(signal['date'], errors='coerce')

#todo: DEPRECATED???: start_date and end_date
#TODO: plot multiple signals by using matplotlib grid plots?

# Function to plot any signal for a given date range
def plot_signal(stock: Equity, signal: ContinuousSignal, start_date=None, end_date=None, cumulative=False,
                order_of_magnitude: str = None, color = 'blue', separator=True) -> int:

    # TODO: plot multiple signals? do it like for events
    # TODO: figure out how to plot percentages
    # Filter the data based on the provided date range
    if start_date is not None or end_date is not None:
        stock.attach(signal) #determines the time window for the signal

    # TODO: can we achieve more isolation so that unwrapping is not needed?
    start_date, end_date = stock.time_window(start_date, end_date)
    ts = signal[start_date : end_date]
    formatter = BigNumberFormatter()
    if cumulative:
        formatter.compute_scaled(ts.cumulative)
    else:
        formatter.compute_scaled(ts.data)
    scaled_signal_filtered = formatter.scaled_data

    plt.plot(ts.time, scaled_signal_filtered, lw=1.0, color=color)
    standard_signal_plot_setup(formatter, start_date, end_date)

    title = f"{signal.label} ({start_date} to {end_date})"
    if cumulative:
        title += " - Cumulative"
    plt.title(title)
    plt.ylabel(f"{signal.label}", color=color)

    if cumulative:
        return signal[-1]
    return signal[-1]

# Function to add a vertical line with a color and corresponding label
def add_vertical_line_with_label(event : Event, color_cycle, linewidth=2):
    # Get the next color in the cycle (we cycle back after all colors are used)
    color = event.color if event.color else color_cycle[len(plt.gca().lines) % len(color_cycle)]
    # Plot the vertical line at the specified date with the color from the cycle
    plt.axvline(x=event.date, color=color, linestyle='--', linewidth=linewidth)
    # Return the label with the color for future reference and date appended at the beginning
    if event.label:
        formatted_date = event.date.strftime("%Y/%m/%d")+" "
        return formatted_date+event.label, color
    return None, None

# Function to add all labels in a box at the top-left of the plot (inside the chart area)
def add_labels_in_box(labels):
    # Access the current axes (the plot area)
    ax = plt.gca()
    # Create a background box in axis coordinates (normalized to the axes)
    ax.annotate('', xy=(0.02, 0.98), xycoords='axes fraction',
                xytext=(0.02, 0.98 - len(labels) * 0.05), textcoords='axes fraction',
                bbox=dict(facecolor='lightyellow', edgecolor='red', boxstyle='round,pad=0.3', lw=2))
    # Add the labels inside the box, formatted neatly
    for i, label_color in enumerate(labels):
        label, color = label_color  # Unpack the label and color
        ax.text(0.02, 0.98 - i * 0.05, label, fontsize=10, ha='left', va='top', color=color, transform=ax.transAxes)

# Function to superpose stock price on the plot, always uses the right-side y-axis
def plot_price(stock : Equity, start_date, end_date, color='red', printOpen=False):
    # Get datetimes and fetch historical price
    start_date, end_date = stock.time_window(start_date, end_date)
    stock_data = stock.get_historical_price(start_date, end_date)
    # Create a secondary y-axis to plot stock price
    ax2 = plt.gca().twinx()
    # Get a good linewidth
    date1, date2 = stock.time_window(start_date, end_date)
    days = (date2 - date1).days
    lw = 2
    if days > 365:
        lw = 1
    # Plot the stock price data (closing price)
    ax2.plot(stock_data['Date'], stock_data['Close'], color='red', label=f"{stock.ticker} Close Price", linestyle='--', linewidth=lw)
    if (printOpen):
        ax2.plot(stock_data['Date'], stock_data['Open'], color='green', label=f"{stock.ticker} Open Price", linestyle='--', linewidth=lw)
    # Set the right y-axis label (price)
    ax2.set_ylabel(f"{stock.ticker} Price (USD)", color=color)
    # Set the y-axis for the price to be more readable
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2f}'))
    # Add a legend for the stock price
    ax2.legend(loc='upper right')
    # title and day separators
    standard_plot_preamble(start_date, end_date)

# Shows events within the time window specified, not inclusive of end_date.
def plot_events(stock : Equity, start_date=None, end_date=None):
    #TODO: could do events that last for a period of time and mix the colours on those bands
    start_date, end_date = stock.time_window(start_date, end_date)
    # Global variable to cycle through colors
    color_cycle = ['#FF0000', '#00AA00', '#0000FF', '#FF7F00', '#800080']  # Red, Green, Blue, Orange, Purple
    # todo: infer date window according to attaching model
    start_date, end_date = stock.time_window(start_date, end_date)
    # Add multiple vertical red lines with labels
    labels = []
    for event in stock.events:
        if (event.date - start_date).days >= 0 > (event.date - end_date).days:
            labels.append(add_vertical_line_with_label(event, color_cycle=color_cycle))
    # Random colors if plotting many events
    if len(labels) > 5:
        r = lambda: random.randint(0, 255)
        for i in range(len(labels)-5):
            color_cycle.append('#%02X%02X%02X' % (r(),r(),r()))
    # Add the labels to the plot inside the box
    add_labels_in_box(labels)
