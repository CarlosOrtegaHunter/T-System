# T-System 🚧 (under construction)

T-System is a lightweight Python-based, object-oriented financial data processing framework designed for historical (and upcoming real-time) market analysis. The design prioritizes flexibility without compromising efficiency.

## 📌 Features
- **Core (`core/`)**: Implements `Equity`, `Options`, `Event`, for structured market data using an object-oriented approach. The class `ContinuousSignal` provides a lazy frame wrapper for time series data.  
- **Analysis (`analysis/`)**: Provides tools for data visualization (`plots.py`) leveraging matplotlib, with support for price, volume, events and signals.

### ContinuousSignal

Particularly, `ContinuousSignal` is designed for efficient operations on timeseries data using polars under the hood while retaining the ease of use of pandas, ideal for large datasets that fit in a personal computer's RAM.

**Key Features:**
- **Lazy Evaluation**: All operations are lazy until explicitly collected, enabling efficient chaining of operations
- **Pandas-like API**: Familiar interface for data manipulation with enhanced performance
- **Arithmetic Operations**: Supports `+`, `-`, `*`, `/`, `//`, `%`, `**` for element-wise operations
- **Date Shifting**: Use `>>` (forward) and `<<` (backward) operators for index-based date shifting
  - `cs >> 35` shifts each observation to the timestamp 35 rows later in the sorted time index
  - `cs << 10` shifts each observation to the timestamp 10 rows earlier
  - For calendar-based shifts, use `timedelta` objects: `cs >> timedelta(days=35)`
- **Date Mapping**: Use `@` operator to map dates based on a mapping table

**Example:**
```python
from core import ContinuousSignal, Equity

GME = Equity("GME", "GameStop")
price_data = GME.get_historical_price("2020-01-01", "2021-06-30")
close_price = ContinuousSignal("GME Close Price", price_data.select(['date', 'close']))

# Shift by 35 trading days (index-based)
t35_shifted = (close_price >> 35).settle("GME Close Price (T+35)")
```

WARNING: Data is primarily fetched from Polygon.io. If a Polygon.io API key is not configured (via environment variable `POLYGON_API_KEY` or `polygon_config.json`), yfinance will be used as a fallback. Note that yfinance has daily rate limits. 

## 📂 Project Structure

```
T-System/
│── analysis/           # Visualization tools
│   ├── formatters.py
│   ├── plots.py
│── common/             # Logging, reading, data processing helpers
│   ├── config.py
│   ├── readers.py
│   ├── utils.py
│── core/               # Market data models
│   ├── Equity.py
│   ├── Options.py
│   ├── Event.py
│   ├── ContinuousSignal.py
:
🚧
:
│── examples/           # Example usage scripts
│   ├── GameStop.py
│
│__ README.md          
```

## Example script GameStop.py 

![image](https://github.com/user-attachments/assets/e76d74f9-ae57-4ca2-a4c6-bc49b9a373c8)
![image](https://github.com/user-attachments/assets/d86fd36b-88e1-433c-8662-12930ecba1ec)
![image](https://github.com/user-attachments/assets/0c273b25-5289-40bc-ab7d-7564c59dc0fc)
![image](https://github.com/user-attachments/assets/c9799645-beef-4e39-8a58-5f8664cb91e2)

## UNDER CONSTRUCTION 
- database integration for OHLCV data. 
- Real-time data streaming. 
- API integration.
- etc

## CLOSED SOURCE
- Evangelion stylized super duper cool UI
- Algo trading engine

---
🚧 **Under construction!**


