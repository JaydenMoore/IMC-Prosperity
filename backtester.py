import csv
import os
import traceback
from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order

class Backtester:
    def __init__(self):
        self.position = {}
        self.pnl_history = []
        self.trader_data = ""
        self.data_dir = os.path.join("round-1-island-data-bottle")  # Relative path
        from main import Trader
        self.trader = Trader()
        self.pnl = {'SQUID_INK': 0.0, 'KELP': 0.0, 'RAINFOREST_RESIN': 0.0}

    def run_csv_backtest(self, file_path: str):
        try:
            # Reset state for new backtest
            products = ['SQUID_INK', 'KELP', 'RAINFOREST_RESIN']
            self.position = {product: 0 for product in products}
            self.pnl = {product: 0.0 for product in products}
            
            is_prices_file = 'prices' in file_path.lower()
            
            with open(file_path, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                headers = [h.strip().lower() for h in next(reader)]
                
                for row in reader:
                    if is_prices_file:
                        if len(row) < 16:  # Skip incomplete rows in prices file
                            continue
                            
                        timestamp = int(row[1])
                        product = row[2]
                        bid_price = float(row[3]) if row[3] else None
                        ask_price = float(row[9]) if row[9] else None
                        
                        if bid_price and ask_price:
                            state = TradingState(
                                timestamp=timestamp,
                                listings={},
                                order_depths={
                                    product: OrderDepth(
                                        buy_orders={int(bid_price): 1},
                                        sell_orders={int(ask_price): 1}
                                    )
                                },
                                position=self.position.copy(),
                                observations={},
                                traderData="",
                                own_trades={},
                                market_trades={}
                            )
                    else:
                        # Process trades file
                        if len(row) < 6:  # Skip incomplete rows in trades file
                            continue
                            
                        timestamp = int(row[0])
                        product = row[3]
                        price = float(row[5])
                        quantity = int(row[6])
                        
                        state = TradingState(
                            timestamp=timestamp,
                            listings={},
                            order_depths={product: OrderDepth(buy_orders={}, sell_orders={})},
                            position=self.position.copy(),
                            observations={},
                            traderData="",
                            own_trades={},
                            market_trades=[{
                                'symbol': product,
                                'price': price,
                                'quantity': quantity,
                                'buyer': row[1],
                                'seller': row[2]
                            }]
                        )
                    
                    # Print trade details for verification
                    if not is_prices_file:
                        print(f"Processing trade: {product} {quantity} @ {price}")
                
                    # Run trader logic
                    orders, conversions, trader_data = self.trader.run(state)
                    
                    # Process orders
                    for product, product_orders in orders.items():
                        for order in product_orders:
                            print(f"{'BUY' if order.quantity > 0 else 'SELL'} {product}: {abs(order.quantity)} @ {order.price}")
                            
                            # Update position and P&L
                            prev_position = self.position.get(product, 0)
                            self.position[product] = prev_position + order.quantity
                            
                            # Calculate P&L impact
                            if order.quantity > 0:  # Buy
                                self.pnl[product] -= abs(order.quantity) * order.price
                            else:  # Sell
                                self.pnl[product] += abs(order.quantity) * order.price
                                
            print('\n=== BACKTEST RESULTS ===')
            print(f'Final Positions: {self.position}')
            print(f'Final P&L:')
            total_pnl = 0
            for product, pnl in self.pnl.items():
                print(f'  {product}: ${pnl:.2f}')
                total_pnl += pnl
            print(f'Total P&L: ${total_pnl:.2f}')
                        
        except Exception as e:
            print(f'[BACKTEST ERROR] {str(e)}')
            traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="Path to CSV file")
    parser.add_argument("--analyze", action="store_true", help="Generate trade analytics")
    args = parser.parse_args()

    backtester = Backtester()
    if args.analyze:
        import pandas as pd
        import numpy as np
        
        # Load data with correct separator
        trades = pd.read_csv(args.file_path, sep=';')
        
        # Key metrics
        stats = {
            'avg_price': trades['price'].mean(),
            'volatility': trades['price'].std(),
            'profit_density': trades['price'].sum() / trades['quantity'].abs().sum(),
            'win_rate': (trades['price'] > 0).mean()
        }
        
        # Save text report
        with open('trade_stats.txt', 'w') as f:
            f.write('=== TRADE ANALYSIS ===\n')
            for k,v in stats.items():
                f.write(f'{k:>15}: {v:.2f}\n')
        
        print(f"Analysis saved to trade_stats.txt")
    else:
        backtester.run_csv_backtest(args.file_path)
