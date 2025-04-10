# Trading Strategy Documentation

## Core Improvements
1. **Dynamic Spread Adjustment**
   - SQUID_INK: Momentum-based (±20% range)
   - KELP: High-frequency tightening (-10% during active periods)

2. **Risk Management**
   - RAINFOREST: Volatility-scaled position limits (10-20 range)
   - Uses 10-period rolling std dev

3. **Performance**
   - Day -2: +5.7% P&L boost
   - More consistent fills across all products

## Backtest Results
| Day | SQUID_INK ($M) | KELP ($M) | RAINFOREST ($M) | Total ($M) |
|-----|----------------|-----------|------------------|------------|
| -2  | 38.9           | 4.3       | 63.1             | 106.3      |
| -1  | 35.5           | 4.0       | 60.0             | 99.5       |
| 0   | 34.3           | 4.1       | 60.0             | 98.4       |

## Submission Checklist
- [x] Final strategy code
- [x] Backtest verification
- [x] Documentation
- [ ] Compression (ZIP) for upload
