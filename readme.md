# T-System 🚧

T-System is a lightweight Python-based, object-oriented financial data processing framework designed for historical (and upcoming real-time) market analysis. The design prioritizes flexibility without compromising efficiency, working on top of polars, and making it easy to extend and integrate with various data sources.

## 📌 Features
- **Core (`core/`)**: Implements `Equity`, `Options`, `Event`, for structured market data using an object-oriented approach. The class `ContinuousSignal` provides a lazy frame wrapper for time series data.  
- **Analysis (`analysis/`)**: Provides tools for data visualization (`plots.py`) leveraging matplotlib, with support for price, volume, events and signals.

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

## UPCOMING
- Date operations and database integration on `ContinuousSignal`.
- Real-time data streaming, enabling continuous market data updates.
- Dash web visualization with historical and real-time updates.
- API integration.
- etc

---
🚧 **Under construction!**