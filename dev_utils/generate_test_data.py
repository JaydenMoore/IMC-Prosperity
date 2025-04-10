import csv
import random
import math
from datetime import datetime, timedelta

# Generate realistic SQUID_INK trade data
def generate_squid_ink_trades(file_path, days_ago=0):
    base_price = 10000
    timestamp = int((datetime.now() - timedelta(days=days_ago)).timestamp() * 1000)
    
    with open(file_path, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['timestamp','buyer','seller','symbol','currency','price','quantity'])
        
        # Mean-reverting pattern parameters
        mean = base_price
        volatility = 0.05
        reversion_speed = 0.1
        price = mean
        
        for _ in range(500):  # 500 trades per day
            timestamp += random.randint(100, 5000)
            
            # Ornstein-Uhlenbeck process
            price = price * (1 - reversion_speed) + \
                    mean * reversion_speed + \
                    random.gauss(0, volatility * mean)
            
            price = max(5000, min(15000, price))
            quantity = random.randint(1, 5)  # Smaller trades
            
            writer.writerow([
                timestamp,
                "",
                "",
                "SQUID_INK",
                "SEASHELLS",
                int(price),
                quantity
            ])
            #start_price = price

# Generate 3 days of test data
for day in [0, 1, 2]:
    generate_squid_ink_trades(f"round-1-island-data-bottle/trades_round_1_day_{-day}.csv", day)
