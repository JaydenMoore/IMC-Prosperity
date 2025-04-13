import csv
import sys
import argparse
import os
from datamodel import OrderDepth, TradingState, Order

class CompetitionBacktester:
    def __init__(self, trader):
        self.trader = trader
        self.position = {}
        self.pnl = {}
    
    def process_row(self, row):
        """Handle both raw strings and pre-parsed lists"""
        try:
            # Handle both string and list inputs
            if isinstance(row, str):
                parts = row.split(';')
            else:
                parts = row
                
            if len(parts) < 15:
                return None
                
            timestamp = int(parts[1])
            product = parts[2]
            
            # Process bids
            buy_orders = {}
            for i in range(3, 9, 2):
                if i+1 < len(parts) and parts[i].strip() and parts[i+1].strip():
                    try:
                        price = float(parts[i])
                        volume = int(parts[i+1])
                        if price > 0 and volume > 0:
                            buy_orders[price] = volume
                    except ValueError:
                        continue
            
            # Process asks
            sell_orders = {}
            for i in range(9, 15, 2):
                if i+1 < len(parts) and parts[i].strip() and parts[i+1].strip():
                    try:
                        price = float(parts[i])
                        volume = int(parts[i+1])
                        if price > 0 and volume > 0:
                            sell_orders[price] = volume
                    except ValueError:
                        continue
            
            return timestamp, product, buy_orders, sell_orders
            
        except Exception as e:
            print(f"Error processing row: {str(e)}", file=sys.stderr)
            return None

    def run(self, file_path):
        """Run backtest with proper error handling"""
        try:
            # Verify file exists
            if not os.path.exists(file_path):
                print(f"Error: Data file not found at {file_path}", file=sys.stderr)
                return
                
            with open(file_path) as f:
                reader = csv.reader(f, delimiter=';')
                next(reader)  # Skip header
                
                for row in reader:
                    result = self.process_row(row)
                    if not result:
                        continue
                        
                    timestamp, product, buy_orders, sell_orders = result
                    
                    # Initialize tracking
                    if product not in self.position:
                        self.position[product] = 0
                        self.pnl[product] = 0.0
                    
                    # Create trading state
                    order_depth = OrderDepth(buy_orders=buy_orders, sell_orders=sell_orders)
                    state = TradingState(
                        timestamp=timestamp,
                        listings={},
                        order_depths={product: order_depth},
                        position=self.position.copy(),
                        observations={},
                        traderData="",
                        own_trades={},
                        market_trades={}
                    )
                    
                    # Run trader logic
                    orders, conversions, trader_data = self.trader.run(state)
                    
                    # Process orders
                    if product in orders:
                        for order in orders[product]:
                            print(f"{'BUY' if order.quantity > 0 else 'SELL'} {product}: {abs(order.quantity)} @ {order.price}")
                            self.position[product] += order.quantity
                            self.pnl[product] -= order.quantity * order.price
                
                # Print final results
                print('\n=== COMPETITION BACKTEST RESULTS ===')
                print(f'Final Positions: {self.position}')
                print('Final P&L:')
                for product, pnl in self.pnl.items():
                    print(f'  {product}: ${pnl:,.2f}')
                print(f'Total P&L: ${sum(self.pnl.values()):,.2f}')
                
                # Calculate arbitrage opportunities captured
                arb_count = sum(1 for order in self.pnl.values() if order != 0)
                print(f'Arbitrage Opportunities Captured: {arb_count}')
                
                # Print position limits compliance
                print('\nPosition Limits Compliance:')
                for product, pos in self.position.items():
                    print(f'  {product}: {pos}/20 ({(pos/20)*100:.1f}% of limit)')
        except Exception as e:
            print(f"Backtest failed: {str(e)}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description='IMC Prosperity Competition Backtester')
    parser.add_argument('--strategy', required=True, help='Path to strategy file')
    parser.add_argument('--data', required=True, help='Path to competition data CSV')
    args = parser.parse_args()
    
    try:
        # Import strategy dynamically
        from importlib.machinery import SourceFileLoader
        strategy = SourceFileLoader('submission', args.strategy).load_module()
        
        # Initialize trader and backtester
        trader = strategy.Trader()
        backtester = CompetitionBacktester(trader)
        
        # Run backtest
        backtester.run(args.data)
    except Exception as e:
        print(f"Backtest failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
