# T-System 🚧 (under construction)

T-System is a lightweight Python-based, object-oriented financial data processing framework designed for historical (and upcoming real-time) market analysis. The design prioritizes flexibility without compromising efficiency.

## 📌 Features
- **Core (`core/`)**: Implements `Equity`, `Options`, `Event`, for structured market data using an object-oriented approach. The class `ContinuousSignal` provides a lazy frame wrapper for time series data.  
- **Analysis (`analysis/`)**: Provides tools for data visualization (`plots.py`) leveraging matplotlib, with support for price, volume, events and signals.

WARNING: For now, it tries to fetch data from Yahoo Finance every time it is executed. If the daily limit is reached, price data might not be shown. 

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
![image](https://github.com/user-attachments/assets/67306f2f-e49c-46ab-88f2-f71b01f81856)

## UPCOMING
- Date operations and database integration on `ContinuousSignal`.
- Real-time data streaming, enabling continuous market data updates.
- Dash web visualization with historical and real-time updates.
- API integration.
- etc

---
🚧 **Under construction!**


