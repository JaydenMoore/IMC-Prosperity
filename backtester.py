"""
IMC Prosperity Backtester - Deterministic Version

Modified to accept --trader argument for comparing different implementations
"""

import argparse
import csv
import os
import traceback
import random
from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order

# Argument parsing
parser = argparse.ArgumentParser(description='IMC Prosperity Backtester')
parser.add_argument('file_path', help='Path to market data CSV file')
parser.add_argument('--trader', default='submission', help='Trader module to use (submission or main)')
parser.add_argument('--analyze', action='store_true', help='Enable detailed trade analysis')
parser.add_argument('--seed', type=int, help='Random seed for deterministic testing')
args = parser.parse_args()

class Backtester:
    def __init__(self, trader_module='submission'):
        self.position = {}
        self.pnl_history = []
        self.trader_data = ""
        self.data_dir = os.path.join("round-1-island-data-bottle")  # Relative path
        self.order_depths = {}
        
        # Dynamic trader import
        if trader_module == 'main':
            from main import Trader
        else:
            from submission import Trader
            
        self.trader = Trader()
        self.pnl = {'SQUID_INK': 0.0, 'KELP': 0.0, 'RAINFOREST_RESIN': 0.0}
        
        if args.seed:
            random.seed(args.seed)
            import numpy as np
            np.random.seed(args.seed)

    def run_csv_backtest(self, file_path: str):
        try:
            # Reset state for new backtest
            products = ['SQUID_INK', 'KELP', 'RAINFOREST_RESIN']
            self.position = {product: 0 for product in products}
            self.pnl = {product: 0.0 for product in products}
            trader_data = ""
            
            with open(file_path) as f:
                reader = csv.reader(f, delimiter=';')
                for row in reader:
                    try:
                        # Handle competition CSV format with validation
                        if len(row) < 6 or not row[5].strip():
                            continue
                            
                        timestamp = int(row[0])
                        product = row[2]
                        price = float(row[5]) if row[5].strip() else 0
                        quantity = int(row[3])
                        
                        if product not in self.order_depths:
                            self.order_depths[product] = OrderDepth()
                        
                        if quantity > 0:  # Buy order
                            if price in self.order_depths[product].buy_orders:
                                self.order_depths[product].buy_orders[price] += quantity
                            else:
                                self.order_depths[product].buy_orders[price] = quantity
                        else:  # Sell order
                            if price in self.order_depths[product].sell_orders:
                                self.order_depths[product].sell_orders[price] += abs(quantity)
                            else:
                                self.order_depths[product].sell_orders[price] = abs(quantity)
                    except Exception as e:
                        print(f"Skipping malformed row: {row}")
                        continue
                    
                    state = TradingState(
                        timestamp=timestamp,
                        listings={},
                        order_depths=self.order_depths,
                        position=self.position.copy(),
                        observations={},
                        traderData=trader_data,
                        own_trades={},
                        market_trades=[]
                    )
                    
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
    backtester = Backtester(args.trader)
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
