import random
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import polars as pl
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator
from analysis.formatters import BigNumberFormatter
from common.config import logger
from core import ContinuousSignal, Event, Equity

class StockPlotter:
    def __init__(self, stock: Equity, start_date=None, end_date=None):
        """
        Initializes the StockPlotter.

        :param stock: The Equity object to plot.
        """
        self.stock = stock
        self.fig, self.ax_left = plt.subplots()
        self.ax_right = None  # Created only when needed
        self.left_used = False  # Track left axis usage
        self.x_axis_is_setup = False
        self.start_date, self.end_date = stock.time_window(start_date, end_date)

    def _setup_x_axis(self, start_date, end_date):
        """Formats the x-axis for date representation."""
        if self.x_axis_is_setup:
            return

        self.ax_left.set_xlabel("Date")
        self.ax_left.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax_left.tick_params(axis="x", rotation=45)

        # Add guiding lines for weekdays
        dates = pd.date_range(start=start_date, end=end_date)
        separator_alpha = 0.2 if len(dates) < 365 else 0.2 * 365 / len(dates)
        for date in dates:
            alpha = separator_alpha * 2 if date.weekday() in [6, 4] else separator_alpha
            self.ax_left.axvline(x=(date + timedelta(hours=12)), color='gray', alpha=alpha, linewidth=0.5)

        self.x_axis_is_setup = True

    def _setup_y_axis(self, ax, formatter: BigNumberFormatter, y_ticks=10):
        """Formats the y-axis and applies big-number formatting."""
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks))
        ax.yaxis.set_major_formatter(formatter.get_formatter())
        ax.yaxis.set_visible(True)

    def _get_right_axis(self):
        """Ensures the right y-axis is created if needed."""
        if self.ax_right is None:
            self.ax_right = self.ax_left.twinx()
        return self.ax_right

    def _hide_unused_axes(self):
        """Hides left and right axes if they are not used."""
        if not self.left_used:
            self.ax_left.yaxis.set_visible(False)
        if self.ax_right and not self.ax_right.has_data():
            self.ax_right.yaxis.set_visible(False)

    def _time_window(self, start, end):
        if not start:
            start = self.start_date
        if not end:
            end = self.end_date
        return start, end

    def plot_volume(self, start_date=None, end_date=None, min_volume=0, exchange="CHX", cumulative=False):
        """Plots exchange trading volume on the left y-axis."""
        start_date, end_date = self._time_window(start_date, end_date)
        data = self.stock.get_historical_volumes(start_date, end_date)
        # todo: test boolean mask works well with cs

        csvolume = ContinuousSignal(f"{exchange} Volume", data.select(pl.col(['date', exchange.lower()])))
        if cumulative:
            csvolume = csvolume.cumulative.settle(f"Cumulative {exchange} Volume")

        x_coord = csvolume.time.collect().to_numpy().flatten()
        y_data = csvolume.data.collect().to_numpy().flatten()
        formatter = BigNumberFormatter(y_data)

        self.ax_left.bar(x_coord, formatter.scaled_data, width=1.0)
        self.left_used = True  # Left axis is now in use

        self._setup_x_axis(start_date, end_date)
        self._setup_y_axis(self.ax_left, formatter)

        self.ax_left.set_title(f"{exchange} Volume ({start_date} to {end_date})" + (" - Cumulative" if cumulative else ""), pad=10)
        self.ax_left.set_ylabel(f"{exchange} Volume (in millions)")

        self._hide_unused_axes()
        #return data[exchange].values[-1]

    def signal(self, signal: ContinuousSignal, start_date=None, end_date=None, cumulative=False, color='blue') -> int:
        """Plots a ContinuousSignal on the left y-axis."""
        start_date, end_date = self._time_window(start_date, end_date)
        ts = signal[start_date: end_date]

        formatter = BigNumberFormatter(signal.data)
        formatter.compute_scaled(ts.cumulative if cumulative else ts.data)
        x_ax = ts.time.collect().to_numpy()
        self.ax_left.plot(x_ax, formatter.scaled_data, lw=1.0, color=color)
        self.left_used = True  # Left axis is now in use

        self._setup_x_axis(start_date, end_date)
        self._setup_y_axis(self.ax_left, formatter)

        self.ax_left.set_title(f"{signal.label} ({start_date} to {end_date})" + (" - Cumulative" if cumulative else ""), pad=10)
        self.ax_left.set_ylabel(f"{signal.label}", color=color)

        self._hide_unused_axes()
        return signal[-1] if cumulative else signal[-1]

    def price(self, start_date=None, end_date=None, color='red', show_open=False):
        """Plots stock price on the right y-axis."""
        start_date, end_date = self._time_window(start_date, end_date)
        stock_data = self.stock.get_historical_price(start_date, end_date)

        ax = self._get_right_axis()
        lw = 2 if (end_date - start_date).days <= 365 else 1
        ax.plot(stock_data['Date'], stock_data['Close'], color=color, linestyle='--', linewidth=lw, label=f"{self.stock.ticker} Close Price")
        if show_open:
            ax.plot(stock_data['Date'], stock_data['Open'], color='green', linestyle='--', linewidth=lw, label=f"{self.stock.ticker} Open Price")

        ax.set_ylabel(f"{self.stock.ticker} Price (USD)", color=color)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2f}'))
        ax.legend(loc='upper right')

        self._setup_x_axis(start_date, end_date)
        self._hide_unused_axes()

    def events(self, start_date=None, end_date=None):
        """Adds vertical lines for stock-related events, with numbering in corresponding colors."""
        start_date, end_date = self._time_window(start_date, end_date)
        color_cycle = ['#FF0000', '#00AA00', '#0000FF', '#FF7F00', '#800080']
        labels = []
        event_positions = {}

        visible_events = [event for event in self.stock.events if start_date <= event.date < end_date]

        for idx, event in enumerate(visible_events, start=1):
            label_info, x_pos = self._add_vertical_line_with_label(event, color_cycle, idx)
            if label_info:
                labels.append((f"({idx}) {label_info[0]}", label_info[1]))  # Prefix number
                event_positions[x_pos] = (idx, label_info[1])  # Store x-coordinates and color

        # Convert axis coordinates to figure coordinates (so numbers are always above)
        transform = self.ax_left.get_xaxis_transform()

        for x_pos, (num, color) in event_positions.items():
            self.ax_left.text(x_pos, 1.002, str(num), fontsize=5, ha='center', va='bottom',
                              color=color, transform=transform, clip_on=False, alpha=0.9)  # Color matches label

        if len(labels) > 5:
            for _ in range(len(labels) - 5):
                r = lambda: random.randint(0, 255)
                color_cycle.append(f'#{r():02X}{r():02X}{r():02X}')

        self._add_labels_in_box(labels)

    def _add_vertical_line_with_label(self, event: Event, color_cycle, event_number):
        """Adds a vertical line for an event, assigns a number, and returns label info."""
        color = event.color or color_cycle[len(self.ax_left.lines) % len(color_cycle)]
        dates = pd.date_range(start=self.start_date, end=self.end_date)
        lw = 1 if len(dates) < 365 else 1 * 365 / len(dates)

        self.ax_left.axvline(x=event.date, color=color, linestyle='--', linewidth=1)

        x_position = event.date  # Store x position for numbering
        label_text = f"{event.date.strftime('%Y/%m/%d')} {event.label}" if event.label else None
        return (label_text, color), x_position

    def _add_labels_in_box(self, labels):
        """Adds numbered event labels inside a box in the top-left of the plot, adapting font size and spacing."""
        ax = self.ax_left
        max_labels = 20  # Upper limit before adjustments start

        # Calculate the available vertical space
        max_box_height = 0.9  # The box can take up to 90% of the plot height
        base_fontsize = 10  # Default font size
        min_fontsize = 5  # Minimum readable font size
        min_spacing = 0.02  # Minimum spacing between lines

        num_labels = len(labels)

        # Adjust both font size and spacing dynamically
        if num_labels > max_labels:
            font_size = max(min_fontsize, base_fontsize * (max_labels / num_labels))
            spacing = max(min_spacing, max_box_height / num_labels)
        else:
            font_size = base_fontsize
            spacing = 0.05  # Default spacing when there is enough space

        # Adjust box height dynamically
        box_height = min(max_box_height, num_labels * spacing)

        # Draw the background box
        ax.annotate('', xy=(0.02, 0.98), xycoords='axes fraction',
                    xytext=(0.02, 0.98 - box_height), textcoords='axes fraction',
                    bbox=dict(facecolor='lightyellow', edgecolor='red', boxstyle='round,pad=0.3', lw=2))

        # Draw the labels with the new font size and spacing
        for i, (label, color) in enumerate(labels):
            ax.text(0.02, 0.98 - i * spacing, label, fontsize=font_size, ha='left', va='top',
                    color=color, transform=ax.transAxes, alpha=0.8)

    def show(self, title=None):
        """Displays the final plot with a tight layout."""
        self.fig.tight_layout()
        self._hide_unused_axes()

        if title:
            self.ax_left.set_title(title, pad=10)

        # TODO: stream the image
        plt.show()
