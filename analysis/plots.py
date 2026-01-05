import base64
import io
import random
from datetime import timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import dash_canvas

import mpld3
import plotly.io as pio
import plotly.graph_objects as go
from dash import html

import polars as pl
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator
from analysis.formatters import BigNumberFormatter
from common.config import logger
from core import ContinuousSignal, Event, Equity

#TODO: live-viz optimizations

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

    def volume(self, start_date=None, end_date=None, min_volume=0, exchange="CHX", cumulative=False):
        """Plots exchange trading volume on the left y-axis."""
        start_date, end_date = self._time_window(start_date, end_date)
        data = self.stock.get_historical_volumes(start_date, end_date)
        # todo: test boolean mask works well with cs
        # todo: show y label when signal is plotted?
        # todo: change min_volume for a lambda filter  

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
        self.ax_left.set_ylabel(f"{exchange} Volume (in millions)" + (" - Cumulative" if cumulative else ""))

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
        self.ax_left.set_ylabel(f"{signal.label}" + (" - Cumulative" if cumulative else ""), color=color)

        self._hide_unused_axes()
        return signal[-1] if cumulative else signal[-1]

    def price(self, start_date=None, end_date=None, color='red', show_open=False):
        """Plots stock price on the right y-axis."""
        start_date, end_date = self._time_window(start_date, end_date)
        stock_data = self.stock.get_historical_price(start_date, end_date)

        ax = self._get_right_axis()
        lw = 2 if (end_date - start_date).days <= 365 else 1
        ax.plot(stock_data['date'], stock_data['close'], color=color, linestyle='--', linewidth=lw, label=f"{self.stock.ticker} Close Price")
        if show_open:
            ax.plot(stock_data['date'], stock_data['open'], color='green', linestyle='--', linewidth=lw, label=f"{self.stock.ticker} Open Price")

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
        plt.show()

    def get_img(self, title=None):
        """
        Converts Matplotlib figure into a Base64-encoded image for Dash.

        :param title: Optional title for the figure.
        :return: Base64-encoded image string.
        """
        self._hide_unused_axes()

        if title:
            self.ax_left.set_title(title, pad=10)

        try:
            # Save Matplotlib figure as PNG
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
            buf.seek(0)

            # Convert PNG to Base64
            encoded_image = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()

            return f"data:image/png;base64,{encoded_image}"

        except Exception as e:
            logger.error(f"ERROR in get_base64_image: {e}")  # Debugging
            return None  # Return None to prevent crashes

    def reset(self):
        """
        Resets the figure by clearing all plots and reinitializing.
        """
        plt.close(self.fig)  # Close the current figure
        self.fig, self.ax_left = plt.subplots()  # Create a new figure
        self.ax_right = None
        self.left_used = False
        self.x_axis_is_setup = False
        logger.info("StockPlotter has been reset!")


    def get_fig(self, title=None):
        """
        Converts Matplotlib figure into a proper Plotly go.Figure without losing details.
        """
        self._hide_unused_axes()

        if title:
            self.ax_left.set_title(title, pad=10)

        # Save Matplotlib figure as PNG
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)

        # Convert PNG to Base64 string
        encoded_image = base64.b64encode(buf.read()).decode("ascii")#("utf-8")
        buf.close()
        fig_bar_matplotlib = f'data:image/png;base64,{encoded_image}'
        return fig_bar_matplotlib
        #
        # # Create a blank Plotly Figure and add Matplotlib as an image overlay
        # fig = go.Figure()
        #
        # fig.add_layout_image(
        #     dict(
        #         source=f"data:image/png;base64,{encoded_image}",
        #         x=0, y=1,  # Align at the top-left corner
        #         xref="paper", yref="paper",
        #         sizex=1, sizey=1,  # Scale to fit
        #         xanchor="left", yanchor="top",
        #         layer="below"  # Ensure it doesn't block other elements
        #     )
        # )
        #
        # # Hide extra gridlines and make sure it looks correct
        # fig.update_layout(
        #     title=title or "Custom Trading Analysis",
        #     xaxis=dict(visible=False),
        #     yaxis=dict(visible=False),
        #     margin=dict(l=0, r=0, t=40, b=0)
        # )
        #
        # return fig  # ✅ Returns a proper go.Figure that actually shows Matplotlib

    #
    # def get_fig_broken(self, title=None):
    #     """
    #     Converts Matplotlib figure into a fully interactive Plotly go.Figure.
    #     :param self:
    #     :param title: Optional title for the figure.
    #     :return: Plotly go.Figure with all Matplotlib traces.
    #     """
    #     self._hide_unused_axes()
    #
    #     if title:
    #         self.ax_left.set_title(title, pad=10)
    #
    #     # Create an empty Plotly figure
    #     fig = go.Figure()
    #
    #     # Loop through all lines in the Matplotlib figure and convert them to Plotly traces
    #     for ax in self.fig.axes:
    #         for line in ax.get_lines():
    #             x_data = line.get_xdata()
    #             y_data = line.get_ydata()
    #
    #             # Extract Matplotlib styling properties
    #             color = line.get_color()
    #             linewidth = line.get_linewidth()
    #             linestyle = line.get_linestyle()
    #             label = line.get_label()
    #
    #             # Convert Matplotlib linestyle to Plotly dash styles
    #             linestyle_map = {
    #                 "-": "solid",
    #                 "--": "dash",
    #                 "-.": "dashdot",
    #                 ":": "dot",
    #                 "None": None
    #             }
    #             plotly_linestyle = linestyle_map.get(linestyle, "solid")
    #
    #             # Add trace with converted styles
    #             fig.add_trace(go.Scatter(
    #                 x=x_data,
    #                 y=y_data,
    #                 mode="lines",
    #                 line=dict(color=color, width=linewidth, dash=plotly_linestyle),
    #                 name=label
    #             ))
    #
    #     # Set figure layout (similar to Matplotlib's axes formatting)
    #     fig.update_layout(
    #         title=title or "Converted Matplotlib Figure",
    #         xaxis=dict(
    #             title="Time",
    #             type="date",
    #             showgrid=True
    #         ),
    #         yaxis=dict(
    #             title="Value",
    #             showgrid=True
    #         ),
    #         template="plotly_white",
    #         margin=dict(l=40, r=40, t=80, b=40)
    #     )
    #
    #     return fig  # ✅ Returns a fully interactive Plotly figure

#
# import plotly.graph_objects as go
# import polars as pl
# import pandas as pd
# from datetime import timedelta
#
# from analysis.formatters import BigNumberFormatter
# from common.config import logger
# from core import ContinuousSignal, Event, Equity
#
# class StockPlotter:
#     def __init__(self, stock: Equity, start_date=None, end_date=None):
#         """
#         Plot volume, signal, price, and events in one Plotly figure.
#         :param stock: The Equity object to plot.
#         """
#         self.stock = stock
#         self.start_date, self.end_date = stock.time_window(start_date, end_date)
#         self.fig = go.Figure()
#
#         # Tracking usage
#         self.left_used = False
#         self.right_used = False
#
#         # Shapes for events + vertical date lines
#         self.shapes = []
#
#         # Optional main title
#         self.title = None
#
#     def _time_window(self, start, end):
#         return start or self.start_date, end or self.end_date
#
#     def _generate_date_lines(self, start_date, end_date):
#         """
#         Vertical lines for each day, with stronger lines for weekends.
#         """
#         dates = pd.date_range(start=start_date, end=end_date, freq='D')
#         # Make lines partially transparent
#         # If too many days, we reduce alpha
#         base_alpha = 0.1 if len(dates) > 365 else 0.2
#         shapes = []
#         for date in dates:
#             # For weekends: double the alpha
#             alpha = base_alpha * 2 if date.weekday() in [5, 6] else base_alpha
#             x_val = date + timedelta(hours=12)
#             shapes.append(dict(
#                 type="line",
#                 x0=x_val, x1=x_val,
#                 y0=0, y1=1,
#                 xref="x", yref="paper",
#                 line=dict(color="gray", width=1),
#                 opacity=alpha,
#                 layer="below"
#             ))
#         return shapes
#
#     def volume(self, start_date=None, end_date=None, exchange="CHX", cumulative=False):
#         """
#         Volume as bold bars on left y-axis (y).
#         """
#         start_date, end_date = self._time_window(start_date, end_date)
#         data = self.stock.get_historical_volumes(start_date, end_date)
#
#         # Retrieve volume data
#         df = data.select(pl.col(["date", exchange.lower()]))
#         x = df["date"].to_pandas()
#         y = df[exchange.lower()].to_pandas()
#
#         if cumulative:
#             y = y.cumsum()
#
#         # Bold bars, distinct color
#         self.fig.add_trace(go.Bar(
#             x=x, y=y,
#             name=f"{exchange} Volume" + (" (Cumulative)" if cumulative else ""),
#             marker_color="rgba(0,0,255,0.7)",
#             yaxis="y",
#             opacity=0.9
#         ))
#         self.left_used = True
#
#     def signal(self, signal: ContinuousSignal, start_date=None, end_date=None, cumulative=False, color='blue') -> int:
#         """
#         Plots a ContinuousSignal on the left y-axis using Plotly.
#         Uses BigNumberFormatter to rescale the signal data.
#
#         :param signal: ContinuousSignal object with signal data and a time component.
#         :param start_date: Optional start date for the data window.
#         :param end_date: Optional end date for the data window.
#         :param cumulative: If True, uses cumulative signal data; otherwise raw data.
#         :param color: Color for the line trace.
#         :return: The last scaled signal value (as an integer).
#         """
#         start_date, end_date = self._time_window(start_date, end_date)
#         ts = signal[start_date: end_date]
#
#         formatter = BigNumberFormatter(signal.data)
#         scaled_data = formatter.compute_scaled(ts.cumulative if cumulative else ts.data)
#         # Use numpy arrays directly
#         x_ax = ts.time.collect().to_numpy()
#
#         self.fig.add_trace(go.Scatter(
#             x=x_ax,
#             y=scaled_data,
#             mode="lines",
#             line=dict(color=color, width=1.0),
#             name=f"{signal.label}" + (" - Cumulative" if cumulative else ""),
#             yaxis="y"
#         ))
#         self.left_used = True
#
#         # Update layout: setting x-axis and y-axis; this replicates your _setup_x_axis and _setup_y_axis
#         self.fig.update_layout(
#             xaxis=dict(
#                 title="Date",
#                 type="date",
#                 tickangle=-45,
#                 showgrid=True
#             ),
#             yaxis=dict(
#                 title=f"{signal.label}" + (" - Cumulative" if cumulative else ""),
#                 titlefont=dict(color=color)
#             )
#         )
#
#         return int(scaled_data[-1])
#
#     def price(self, start_date=None, end_date=None, color="red", show_open=False):
#         """
#         Plot stock price on the right axis (y2) as a bold dashed line.
#         """
#         start_date, end_date = self._time_window(start_date, end_date)
#         data = self.stock.get_historical_price(start_date, end_date)
#
#         x = data["date"].to_pandas()
#         close = data["close"].to_pandas()
#
#         self.fig.add_trace(go.Scatter(
#             x=x, y=close,
#             mode="lines",
#             line=dict(color=color, width=3, dash="dash"),
#             name=f"{self.stock.ticker} Close Price",
#             yaxis="y2"
#         ))
#         self.right_used = True
#
#         if show_open and "open" in data.columns:
#             open_prices = data["open"].to_pandas()
#             self.fig.add_trace(go.Scatter(
#                 x=x, y=open_prices,
#                 mode="lines",
#                 line=dict(color="green", width=2, dash="dot"),
#                 name=f"{self.stock.ticker} Open Price",
#                 yaxis="y2"
#             ))
#
#     def events(self, start_date=None, end_date=None):
#         """
#         Adds vertical event lines and text annotations to the Plotly figure.
#         The lines and labels are superposed over the main plot.
#         """
#         start_date, end_date = self._time_window(start_date, end_date)
#         color_cycle = ['#FF0000', '#00AA00', '#0000FF', '#FF7F00', '#800080']
#         visible_events = [event for event in self.stock.events if start_date <= event.date < end_date]
#
#         shapes = []
#         annotations = []
#         for idx, event in enumerate(visible_events, start=1):
#             color = event.color or color_cycle[(idx - 1) % len(color_cycle)]
#             # Create a vertical line for the event
#             shapes.append({
#                 'type': 'line',
#                 'x0': event.date,
#                 'x1': event.date,
#                 'y0': 0,
#                 'y1': 1,
#                 'xref': 'x',
#                 'yref': 'paper',
#                 'line': {'color': color, 'width': 2, 'dash': 'dash'},
#                 'layer': 'above'
#             })
#             # Add an annotation just above the plot for the event
#             annotations.append({
#                 'x': event.date,
#                 'y': 1.05,
#                 'xref': 'x',
#                 'yref': 'paper',
#                 'text': f"({idx}) {event.label}" if event.label else f"({idx})",
#                 'showarrow': False,
#                 'font': {'color': color, 'size': 12}
#             })
#         # Update the figure layout with the new shapes and annotations
#         self.fig.update_layout(shapes=shapes, annotations=annotations)
#         return self.fig
#
#     def show(self, title="Test"):
#         """
#         Finalize everything, overlay left + right axes, add date lines, and return the figure.
#         """
#         try:
#             # Date lines
#             date_shapes = self._generate_date_lines(self.start_date, self.end_date)
#             self.shapes.extend(date_shapes)
#             self.fig.update_layout(shapes=self.shapes)
#
#             # Left axis config
#             if self.left_used:
#                 self.fig.update_layout(
#                     yaxis=dict(
#                         title="Volume / Signals",
#                         side="left",
#                         showgrid=True,
#                         overlaying="free",
#                         visible=True
#                     )
#                 )
#             else:
#                 self.fig.update_layout(yaxis=dict(visible=False))
#
#             # Right axis config
#             if self.right_used:
#                 self.fig.update_layout(
#                     yaxis2=dict(
#                         title=f"{self.stock.ticker} Price (USD)",
#                         side="right",
#                         overlaying="y",
#                         showgrid=False,
#                         visible=True,
#                         position=1.0
#                     )
#                 )
#             else:
#                 self.fig.update_layout(yaxis2=dict(visible=False))
#
#             self.fig.update_layout(
#                 title=dict(text=title, font=dict(size=24)),
#                 xaxis=dict(
#                     title="Date",
#                     type="date",
#                     showgrid=True,
#                     tickangle=-45,
#                     tickfont=dict(size=12)
#                 ),
#                 template="plotly_white",
#                 legend=dict(
#                     orientation="h",
#                     yanchor="bottom", y=-0.2,
#                     xanchor="center", x=0.5,
#                     font=dict(size=12)
#                 ),
#                 margin=dict(l=50, r=50, t=70, b=70)
#             )
#             return self.fig
#         except Exception as e:
#             logger.error(f"Error in show(): {e}", exc_info=True)
#             return go.Figure()
