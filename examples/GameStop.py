import os
import sys

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.plots import *
from core import Equity, Event, ContinuousSignal

# CREATE UNIQUE EQUITY INSTANCE
GME = Equity("GME", "GameStop")

# POPULATE EQUITY VIEW WITH EVENTS
GME_PCO = Event("2021-01-28", label="Sneeze PCO", color='black')
split_dividend = Event("2022-07-22", label="4:1 Split dividend", color='green')
GME.attach([GME_PCO, split_dividend])

RC_initial = Event("2020-08-28", label="RC Ventures acquires 5.8m shares (23m post split)", color='blue')
RC_9m = Event("2020-12-17", label="RC Ventures holding 9m shares (36m post split)", color='blue')
RC_letter = Event("2020-11-16", label="RC Ventures Letter to the board", color='blue')
RC_joins_board = Event("2021-01-11", label="Ryan Cohen joins GameStop's board", color='blue')
RC_chairman = Event("2021-06-09", label="Ryan Cohen becomes chairman", color='blue')
RC_CEO = Event("2023-09-23", label="Ryan Cohen becomes CEO", color='blue')
RC_transfer = Event("2025-01-29", label="Ryan Cohen's ownership restructure", color='blue')
GME.attach([RC_initial, RC_9m, RC_letter, RC_joins_board, RC_chairman, RC_CEO, RC_transfer])

DFV_ddown50 = Event("2021-01-05", label="DFV YOLO 10k->50k shares", color='orangered')
DFV_jan_expiry = Event("2021-01-15", label="DFV calls almost expiring worthless!", color='orangered')
DFV_testified = Event("2021-02-17", label="DFV testifies to congress", color='orangered')
DFV_ddown100 = Event("2021-02-19", label="DFV YOLO doubles down 50k->100k sh", color='orangered')
DFV_ddown200 = Event("2021-04-16", label="DFV YOLO final update (->200k sh)", color='orangered')
GME.attach([DFV_ddown50, DFV_jan_expiry, DFV_testified, DFV_ddown100, DFV_ddown200])

DFV_RunLolaRun = Event("2024-05-09", label="DFV Run Lola Run", color='orangered')
DFV_comeback = Event("2024-06-02", label="DFV comeback YOLO (5m sh, 120k c)", color='orangered')
DFV_comeback2 = Event("2024-06-03", label="DFV comeback YOLO (5m sh, 120k c)", color='orangered')
DFV_comeback3 = Event("2024-06-06", label="DFV comeback YOLO (5m sh, 120k c)", color='orangered')
DFV_stream = Event("2024-06-07", label="DFV YouTube stream (5m sh, 120k c)", color='orangered')
DFV_comeback4 = Event("2024-06-10", label="DFV comeback YOLO (5m sh, 120k c)", color='orangered')
DFV_comeback5 = Event("2024-06-13", label="DFV comeback YOLO (9m shares)", color='orangered')
GME.attach([DFV_RunLolaRun, DFV_comeback, DFV_comeback2, DFV_comeback3, DFV_stream, DFV_comeback4, DFV_comeback5])

########################################################################
# EXAMPLE: RYAN COHEN'S ACQUISITION SHOWING IN CHICAGO EXCHANGE VOLUME #
########################################################################

# Chicago Exchange Volume seems to be representative of Ryan's acquisition
# of GameStop shares.

sdate = "2020-03-01"
edate = "2021-01-20"

plot = StockPlotter(GME, sdate, edate)
plot.price()
plot.volume(exchange="CHX", cumulative=True)
plot.events()
plot.show("Ryan Cohen's acquisition")


########################################################################
#          EXAMPLE: ANALYZE MARKET DATA WITH PUBLIC INFORMATION        #
########################################################################

plot = StockPlotter(GME, "2024-04-01", "2024-08-20")
plot.price()
plot.volume(exchange="CHX")
plot.events()
plot.signal(GME.Options['Call Volume'])
plot.show("Roaring Kitty's return (Call Volume and CHX Volume)")

########################################################################
#                  EXAMPLE: OPERATIONS WITH SIGNALS                    #
########################################################################

#Create a bullish signal!

bullish_signal = GME.Options['Imp Vol'] * (GME.Options['Call OI'] - GME.Options['Put OI'])

plot = StockPlotter(GME, "2020-10-01", "2021-04-20")
plot.price()
plot.signal(bullish_signal.settle("Bullish Relative OI weighted by Imp Vol"))
plot.events()
plot.show("Bullish Options Indicator")

########################################################################
#          EXAMPLE: USE SLICES TO OPERATE ON DATE RANGES               #
########################################################################

call_OI = GME.Options['Call OI']
call_OI[:split_dividend.date] *= 4

plot = StockPlotter(GME, "2019-01-01")
plot.price()
plot.events()
plot.signal(call_OI)
plot.show("Historical GME & Call OI (split-adjusted retroactively)")


#print(call_OI.time.collect().to_numpy())


#print((call_OI[sdate:edate] << timedelta(days=100)).time.collect().to_numpy())

#1. get item
# 2. lshift

########################################################################
#          EXAMPLE: T+35 FTD CYCLE ANALYSIS USING DATE SHIFTING        #
########################################################################

# T+35 FTD cycle: SEC requires FTD positions to be closed within 35 trading days.
# Here, >> 35 shifts by 35 *rows* in the time index (trading days for daily OHLCV).

price_data = GME.get_historical_price("2020-01-01", "2021-06-30")
close_price = ContinuousSignal("GME Close Price", price_data.select(['date', 'close']))
t35_shifted = (close_price >> 35).settle("GME Close Price (T+35)")

plot = StockPlotter(GME, "2020-03-01", "2021-06-30")
plot.price()
plot.signal(close_price, start_date="2020-03-01", end_date="2021-06-30")
plot.signal(t35_shifted, color='orange', start_date="2020-03-01", end_date="2021-06-30")
plot.events()
plot.show("T+35 FTD Cycle Analysis")
