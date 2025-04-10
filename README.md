# IMC Prosperity Round 1 Submission

Official submission for IMC Prosperity Challenge Round 1 - Island Trading

## Contents
- `main.py`: Optimized trading algorithm
- `strategy_notes.md`: Performance documentation
- `backtester.py`: Custom backtesting engine
- `/round-1-island-data-bottle`: Market data

## Project Structure
```
.
├── main.py                # Core trading algorithm
├── backtester.py          # Backtesting engine
├── datamodel.py           # Data structures
├── modules.py             # Shared utilities
├── strategy_notes.md      # Performance documentation
├── requirements.txt       # Python dependencies
└── round-1-island-data-bottle/
    ├── prices_round_1_day_-2.csv
    ├── prices_round_1_day_-1.csv
    ├── prices_round_1_day_0.csv
    ├── trades_round_1_day_-2.csv
    ├── trades_round_1_day_-1.csv
    └── trades_round_1_day_0.csv
```

## Key Components
1. **Trading Algorithm** (`main.py`)
   - Implements product-specific strategies
   - Handles order generation and risk management

2. **Backtesting System** (`backtester.py`)
   - Processes market data files
   - Simulates exchange interactions
   - Generates P&L reports

3. **Market Data**
   - 3 days of price/trade data in CSV format
   - Paths are relative for portability

## Requirements
- Python 3.8+
- pandas, numpy

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Run backtest
python backtester.py round-1-island-data-bottle/prices_round_1_day_-2.csv

# Run tests
python -m unittest unit_test.py
python run_tests.py
```

## Key Features
- Dynamic spread adjustment
- Volatility-scaled position limits
- Momentum-based trading

## Testing
Test coverage includes:
- Unit tests (`unit_test.py`)
- Integration tests (`run_tests.py`)
- New trader scenarios (`new_trader_tests.py`)

## Path Conventions
- All paths are relative to project root
- Uses `os.path.join()` for cross-platform compatibility
- Data files in `round-1-island-data-bottle/`

## Repository Metadata
Description: IMC Prosperity Round 1 - Optimized Island Trading Algorithm
Topics: algorithmic-trading, market-making, imc-prosperity
