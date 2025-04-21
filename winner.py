import numpy as np
import math
import json
from datamodel import OrderDepth, TradingState, Order, Observation, ConversionObservation
from typing import Dict, List, Any, Tuple

# Helper function for Normal CDF using math.erf
def _norm_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

# Helper function for Normal PDF
def _norm_pdf(x):
    """Probability density function for the standard normal distribution."""
    return math.exp(-x**2 / 2.0) / math.sqrt(2.0 * math.pi)

class Trader:
    def __init__(self):
        self.position_limit = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "MAGNIFICENT_MACARONS": 75,
        }
        self.max_volume = { # Example max order volumes, adjust as needed
            "RAINFOREST_RESIN": 10, "KELP": 10, "SQUID_INK": 10,
            "CROISSANTS": 50, "JAMS": 50, "DJEMBES": 10,
            "PICNIC_BASKET1": 10, "PICNIC_BASKET2": 20,
            "VOLCANIC_ROCK": 40,
            "VOLCANIC_ROCK_VOUCHER_9500": 20, "VOLCANIC_ROCK_VOUCHER_9750": 20,
            "VOLCANIC_ROCK_VOUCHER_10000": 20, "VOLCANIC_ROCK_VOUCHER_10250": 20,
            "VOLCANIC_ROCK_VOUCHER_10500": 20,
            "MAGNIFICENT_MACARONS": 10,
        }
        self.price_history: Dict[str, List[float]] = {}
        self.ma_length = 20  # Moving average window
        self.squid_deviation_threshold = 5.0  # Threshold for Squid Ink reversion
        self.picnic_arb_margin = 50.0 # Min profit for picnic basket arbitrage
        self.volcanic_risk_free_rate = 0.0 # Assuming 0 risk-free rate
        self.volcanic_sigma_guess = 0.2 # Initial guess for volatility (needs calibration)
        self.voucher_strikes = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500, "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000, "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }
        self.macaron_csi_threshold = 2500 # Critical Sunlight Index threshold (needs calibration)
        self.macaron_storage_cost_per_unit_per_tick = 0.1
        self.macaron_conversion_limit = 10
        self.last_sunlight = None
        self.last_mid_price: Dict[str, float] = {} # Cache last mid price

    def _get_mid_price(self, product: str, order_depth: OrderDepth) -> float | None:
        """Safely calculate the mid-price from the order book."""
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return self.last_mid_price.get(product, None) # Fallback to last known
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        self.last_mid_price[product] = mid_price # Update cache
        return mid_price

    def _get_best_bid_ask(self, order_depth: OrderDepth) -> Tuple[float | None, float | None]:
         """Safely gets best bid and ask."""
         best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
         best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
         return best_bid, best_ask

    def _update_price_history(self, product: str, mid_price: float):
        """Update price history for a product."""
        if mid_price is None: return
        if product not in self.price_history:
            self.price_history[product] = []
        self.price_history[product].append(mid_price)
        # Keep history length manageable, e.g., 100 + ma_length
        max_hist_len = max(self.ma_length + 1, 100)
        self.price_history[product] = self.price_history[product][-max_hist_len:]

    def _calculate_sma(self, product: str, window: int) -> float | None:
        """Calculate Simple Moving Average."""
        if product in self.price_history and len(self.price_history[product]) >= window:
            return np.mean(self.price_history[product][-window:])
        return None

    def _calculate_std_dev(self, product: str, window: int) -> float | None:
        """Calculate Standard Deviation."""
        if product in self.price_history and len(self.price_history[product]) >= window:
            return np.std(self.price_history[product][-window:])
        return None

    def _compute_order_volume(self, product: str, side: str, current_pos: int, base_volume: int = 0) -> int:
        """Computes order volume considering position limits."""
        pos_limit = self.position_limit.get(product, 0)
        max_vol = self.max_volume.get(product, 10) # Use product specific max vol
        if base_volume == 0:
            base_volume = max_vol # Default to max_vol if not specified

        if side == "BUY":
            available_limit = pos_limit - current_pos
        elif side == "SELL":
            available_limit = pos_limit + current_pos # Available space for shorting
        else:
            return 0 # Invalid side

        if available_limit <= 0:
            return 0

        volume = min(base_volume, available_limit)
        return max(0, int(volume)) # Ensure non-negative integer

    # --- Black-Scholes Helper Functions ---
    def _black_scholes_call(self, S, K, T, r, sigma):
        """Calculate Black-Scholes call option price."""
        if T <= 0 or sigma <= 0: return max(0, S - K) # Intrinsic value at expiry or if no vol
        # Check for potential division by zero or log of non-positive number
        if S <= 0 or K <= 0: return max(0, S - K * math.exp(-r * T)) # Handle edge cases
        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            call_price = (S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2))
        except (ValueError, OverflowError):
             # Handle potential math errors (e.g., log of zero, overflow)
             # Fallback to intrinsic value or some other safe value
             call_price = max(0, S - K * math.exp(-r * T))
        return call_price

    def _implied_volatility(self, S, K, T, r, market_price, sigma_guess=0.2, max_iter=100, tol=1e-5):
        """Calculate implied volatility using Newton-Raphson method."""
        sigma = sigma_guess
        for _ in range(max_iter):
            try:
                price = self._black_scholes_call(S, K, T, r, sigma)
                vega = self._bs_vega(S, K, T, r, sigma)
            except (ValueError, OverflowError):
                # If BS calculation fails, cannot proceed with IV calculation
                return sigma_guess # Return initial guess or some default

            if vega == 0: return sigma # Avoid division by zero, return current sigma

            diff = price - market_price
            if abs(diff) < tol:
                return sigma

            # Newton-Raphson step
            sigma = sigma - diff / vega

            # Ensure sigma stays positive and within reasonable bounds
            if sigma <= 0:
                sigma = tol # Reset to a small positive value
            elif sigma > 5.0: # Add an upper bound check if needed
                 sigma = 5.0 # Cap volatility if it explodes

        # If not converged, return the last valid sigma or the initial guess
        return sigma if sigma > 0 else sigma_guess

    def _bs_vega(self, S, K, T, r, sigma):
        """Calculate Vega for Black-Scholes."""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0
        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            vega = S * _norm_pdf(d1) * math.sqrt(T)
        except (ValueError, OverflowError):
            vega = 0 # Handle potential math errors
        return vega

    def _get_tte(self, timestamp: int) -> float:
        """Estimate Time To Expiry (TTE) in years. Assumes 1 round = 1 day, 1 day = 1/252 years."""
        # Total duration is 7 days = 7 * 100 * 100 = 70000 timestamps
        # Round 1 starts at 0, ends near 100000 (1 day)
        # Round 5 ends near 500000 (5 days) -> 2 days left
        total_trading_days = 7
        ticks_per_day = 100000 # 100 ticks/second * 1000 seconds
        total_ticks = total_trading_days * ticks_per_day

        remaining_ticks = total_ticks - timestamp
        remaining_days = remaining_ticks / ticks_per_day
        tte_years = remaining_days / 252.0 # Assuming 252 trading days/year
        return max(1e-6, tte_years) # Avoid T=0 before expiry

    # --- Product Handlers ---

    def _handle_rainforest_resin(self, order_depth: OrderDepth, position: int) -> List[Order]:
        """Stable market making for Rainforest Resin."""
        orders = []
        best_bid, best_ask = self._get_best_bid_ask(order_depth)
        max_vol = self.max_volume["RAINFOREST_RESIN"]

        if best_bid is not None and best_ask is not None:
            # Place buy order slightly below best ask
            buy_price = best_ask - 1
            buy_volume = self._compute_order_volume("RAINFOREST_RESIN", "BUY", position, max_vol)
            if buy_volume > 0:
                orders.append(Order("RAINFOREST_RESIN", buy_price, buy_volume))

            # Place sell order slightly above best bid
            sell_price = best_bid + 1
            sell_volume = self._compute_order_volume("RAINFOREST_RESIN", "SELL", position, max_vol)
            if sell_volume > 0:
                orders.append(Order("RAINFOREST_RESIN", sell_price, -sell_volume))
        return orders

    def _handle_kelp(self, order_depth: OrderDepth, position: int) -> List[Order]:
        """Trend-following strategy for Kelp using SMA."""
        orders = []
        mid_price = self._get_mid_price("KELP", order_depth)
        self._update_price_history("KELP", mid_price)
        moving_average = self._calculate_sma("KELP", self.ma_length)
        max_vol = self.max_volume["KELP"]

        if mid_price is None or moving_average is None:
            return orders

        best_bid, best_ask = self._get_best_bid_ask(order_depth)

        if mid_price > moving_average + 0.5:  # Uptrend signal (added buffer)
            # Aggressively buy near best bid or slightly above
            if best_bid is not None:
                buy_price = best_bid + 1
                buy_volume = self._compute_order_volume("KELP", "BUY", position, max_vol)
                if buy_volume > 0:
                    orders.append(Order("KELP", buy_price, buy_volume))
        elif mid_price < moving_average - 0.5:  # Downtrend signal (added buffer)
            # Aggressively sell near best ask or slightly below
            if best_ask is not None:
                sell_price = best_ask - 1
                sell_volume = self._compute_order_volume("KELP", "SELL", position, max_vol)
                if sell_volume > 0:
                    orders.append(Order("KELP", sell_price, -sell_volume))
        # Optional: Add passive MM orders if no trend signal
        # else: ... add passive orders ...

        return orders

    def _handle_squid_ink(self, order_depth: OrderDepth, position: int) -> List[Order]:
        """Mean-reversion strategy for Squid Ink."""
        orders = []
        mid_price = self._get_mid_price("SQUID_INK", order_depth)
        self._update_price_history("SQUID_INK", mid_price)
        moving_average = self._calculate_sma("SQUID_INK", self.ma_length)
        max_vol = self.max_volume["SQUID_INK"]

        if mid_price is None or moving_average is None:
            return orders

        deviation = mid_price - moving_average
        best_bid, best_ask = self._get_best_bid_ask(order_depth)

        # Sell if price is significantly above average
        if deviation > self.squid_deviation_threshold:
            if best_bid is not None:
                sell_price = best_bid # Hit the bid for reversion
                sell_volume = self._compute_order_volume("SQUID_INK", "SELL", position, max_vol)
                 # Scale volume by deviation?
                # sell_volume = int(sell_volume * min(1.5, deviation / self.squid_deviation_threshold))
                # sell_volume = min(sell_volume, order_depth.buy_orders.get(sell_price, 0)) # Consider liquidity
                if sell_volume > 0:
                    orders.append(Order("SQUID_INK", sell_price, -sell_volume))

        # Buy if price is significantly below average
        elif deviation < -self.squid_deviation_threshold:
            if best_ask is not None:
                buy_price = best_ask # Hit the ask for reversion
                buy_volume = self._compute_order_volume("SQUID_INK", "BUY", position, max_vol)
                # Scale volume by deviation?
                # buy_volume = int(buy_volume * min(1.5, abs(deviation) / self.squid_deviation_threshold))
                # buy_volume = min(buy_volume, -order_depth.sell_orders.get(buy_price, 0)) # Consider liquidity
                if buy_volume > 0:
                    orders.append(Order("SQUID_INK", buy_price, buy_volume))
        # No passive MM to avoid risk in high volatility as per hint
        return orders

    def _handle_picnic_baskets(self, state: TradingState) -> List[Order]:
        """Arbitrage logic for PICNIC_BASKET1, PICNIC_BASKET2 and components."""
        orders = []
        baskets = ["PICNIC_BASKET1", "PICNIC_BASKET2"]
        components = ["CROISSANTS", "JAMS", "DJEMBES"]
        basket_recipes = {
            "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
            "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}
        }
        all_products = baskets + components

        # Ensure all necessary order depths are present
        if not all(p in state.order_depths for p in all_products):
            # print("Warning: Missing order depth for picnic basket/components")
            return orders # Cannot proceed if any book is missing

        # Get best bid/ask for all involved products
        prices = {}
        for p in all_products:
            bb, ba = self._get_best_bid_ask(state.order_depths[p])
            if bb is None or ba is None:
                # print(f"Warning: Missing bid/ask for {p}")
                return orders # Cannot proceed if any price is missing
            prices[p] = {'bid': bb, 'ask': ba}

        # --- Arbitrage Logic ---
        for basket_name, recipe in basket_recipes.items():
            # Calculate cost to buy components (using asks)
            cost_to_buy_components = 0
            for comp, qty in recipe.items():
                cost_to_buy_components += qty * prices[comp]['ask']

            # Calculate revenue from selling components (using bids)
            revenue_from_selling_components = 0
            for comp, qty in recipe.items():
                revenue_from_selling_components += qty * prices[comp]['bid']

            basket_bid = prices[basket_name]['bid']
            basket_ask = prices[basket_name]['ask']

            # Arb Opportunity 1: Buy components, Sell basket
            profit1 = basket_bid - cost_to_buy_components
            if profit1 > self.picnic_arb_margin:
                # Determine max volume based on limits and liquidity
                vol = self.max_volume[basket_name] # Start with max basket volume
                # Check basket sell limit
                vol = min(vol, self._compute_order_volume(basket_name, "SELL", state.position.get(basket_name, 0)))
                # Check component buy limits (converted to basket units)
                for comp, qty in recipe.items():
                    comp_vol_limit = self._compute_order_volume(comp, "BUY", state.position.get(comp, 0))
                    vol = min(vol, comp_vol_limit // qty if qty > 0 else vol)
                # Check liquidity (simplified: check best price liquidity)
                vol = min(vol, state.order_depths[basket_name].buy_orders.get(basket_bid, 0))
                for comp, qty in recipe.items():
                    comp_liq = -state.order_depths[comp].sell_orders.get(prices[comp]['ask'], 0) # Sell orders are negative
                    vol = min(vol, comp_liq // qty if qty > 0 else vol)

                vol = int(max(0, vol))
                if vol > 0:
                    print(f"ARB: Buy {list(recipe.keys())}, Sell {basket_name}. Profit: {profit1:.2f}, Vol: {vol}")
                    orders.append(Order(basket_name, basket_bid, -vol))
                    for comp, qty in recipe.items():
                        orders.append(Order(comp, prices[comp]['ask'], vol * qty))

            # Arb Opportunity 2: Buy basket, Sell components
            profit2 = revenue_from_selling_components - basket_ask
            if profit2 > self.picnic_arb_margin:
                 # Determine max volume based on limits and liquidity
                vol = self.max_volume[basket_name]
                # Check basket buy limit
                vol = min(vol, self._compute_order_volume(basket_name, "BUY", state.position.get(basket_name, 0)))
                # Check component sell limits
                for comp, qty in recipe.items():
                    comp_vol_limit = self._compute_order_volume(comp, "SELL", state.position.get(comp, 0))
                    vol = min(vol, comp_vol_limit // qty if qty > 0 else vol)
                # Check liquidity
                vol = min(vol, -state.order_depths[basket_name].sell_orders.get(basket_ask, 0))
                for comp, qty in recipe.items():
                    comp_liq = state.order_depths[comp].buy_orders.get(prices[comp]['bid'], 0)
                    vol = min(vol, comp_liq // qty if qty > 0 else vol)

                vol = int(max(0, vol))
                if vol > 0:
                    print(f"ARB: Buy {basket_name}, Sell {list(recipe.keys())}. Profit: {profit2:.2f}, Vol: {vol}")
                    orders.append(Order(basket_name, basket_ask, vol))
                    for comp, qty in recipe.items():
                        orders.append(Order(comp, prices[comp]['bid'], -vol * qty))

        # --- Market Making for Components (if not involved in arb) ---
        # Basic MM for components - can reuse rainforest logic or make specific
        arb_traded_products = {o.symbol for o in orders}
        for p in components + baskets:
             if p not in arb_traded_products and p in state.order_depths:
                 # Apply basic MM (like rainforest)
                 pos = state.position.get(p, 0)
                 od = state.order_depths[p]
                 bb, ba = self._get_best_bid_ask(od)
                 max_vol = self.max_volume.get(p, 10)
                 if bb is not None and ba is not None:
                     buy_price = min(ba - 1, bb + 1) # Adjust pennying logic maybe
                     buy_volume = self._compute_order_volume(p, "BUY", pos, max_vol // 2) # Smaller MM volume
                     if buy_volume > 0 and buy_price < ba: # Ensure valid price
                         orders.append(Order(p, buy_price, buy_volume))

                     sell_price = max(bb + 1, ba - 1) # Adjust pennying
                     sell_volume = self._compute_order_volume(p, "SELL", pos, max_vol // 2)
                     if sell_volume > 0 and sell_price > bb: # Ensure valid price
                         orders.append(Order(p, sell_price, -sell_volume))

        return orders

    def _handle_volcanic_rock(self, order_depth: OrderDepth, position: int) -> List[Order]:
        """Market making or trend following for Volcanic Rock."""
        # Using simple MM for now, similar to Rainforest Resin
        orders = []
        best_bid, best_ask = self._get_best_bid_ask(order_depth)
        max_vol = self.max_volume["VOLCANIC_ROCK"]

        if best_bid is not None and best_ask is not None:
            # Place buy order
            buy_price = best_ask - 1
            buy_volume = self._compute_order_volume("VOLCANIC_ROCK", "BUY", position, max_vol)
            if buy_volume > 0:
                orders.append(Order("VOLCANIC_ROCK", buy_price, buy_volume))

            # Place sell order
            sell_price = best_bid + 1
            sell_volume = self._compute_order_volume("VOLCANIC_ROCK", "SELL", position, max_vol)
            if sell_volume > 0:
                orders.append(Order("VOLCANIC_ROCK", sell_price, -sell_volume))
        return orders

    def _handle_volcanic_vouchers(self, state: TradingState) -> List[Order]:
        """Options strategy for Volcanic Rock Vouchers."""
        orders = []
        underlying = "VOLCANIC_ROCK"
        vouchers = list(self.voucher_strikes.keys())

        if underlying not in state.order_depths: return orders
        S = self._get_mid_price(underlying, state.order_depths[underlying])
        if S is None or S <= 0: return orders # Ensure S is positive

        T = self._get_tte(state.timestamp)
        r = self.volcanic_risk_free_rate
        # Simple volatility: historical vol of underlying
        # sigma_hist = self._calculate_std_dev(underlying, 50) # Use recent std dev
        # Using fixed guess as historical calculation needs careful annualization
        sigma = self.volcanic_sigma_guess

        # --- Strategy: Compare BS Theoretical Price vs Market Price ---
        for voucher in vouchers:
            if voucher not in state.order_depths: continue
            K = self.voucher_strikes[voucher]
            if K <= 0: continue # Ensure K is positive

            order_depth = state.order_depths[voucher]
            market_bid, market_ask = self._get_best_bid_ask(order_depth)
            market_mid = self._get_mid_price(voucher, order_depth)
            position = state.position.get(voucher, 0)
            max_vol = self.max_volume[voucher]

            if market_bid is None or market_ask is None or market_mid is None: continue

            # Calculate theoretical value
            theoretical_price = self._black_scholes_call(S, K, T, r, sigma)

            # Trading Signal (Buy if cheap, Sell if expensive) - Adjust thresholds as needed
            # Using a fixed spread instead of percentage might be better for low priced options
            spread_threshold = 2.0 # Example: Trade if market is $2 away from theoretical
            buy_threshold = theoretical_price - spread_threshold
            sell_threshold = theoretical_price + spread_threshold

            # Place Orders
            if market_ask < buy_threshold:
                buy_volume = self._compute_order_volume(voucher, "BUY", position, max_vol)
                # Consider liquidity at ask
                liq = -order_depth.sell_orders.get(market_ask, 0) if market_ask in order_depth.sell_orders else 0
                final_volume = min(buy_volume, liq)
                if final_volume > 0:
                    print(f"BUY {voucher}: {final_volume} @ {market_ask} (Theo: {theoretical_price:.2f})")
                    orders.append(Order(voucher, market_ask, final_volume))

            elif market_bid > sell_threshold:
                sell_volume = self._compute_order_volume(voucher, "SELL", position, max_vol)
                 # Consider liquidity at bid
                liq = order_depth.buy_orders.get(market_bid, 0) if market_bid in order_depth.buy_orders else 0
                final_volume = min(sell_volume, liq)
                if final_volume > 0:
                    print(f"SELL {voucher}: {final_volume} @ {market_bid} (Theo: {theoretical_price:.2f})")
                    orders.append(Order(voucher, market_bid, -final_volume))

            # Optional: Delta hedging
            # try:
            #     d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            #     delta = _norm_cdf(d1)
            #     # ... implement hedging logic based on total delta ...
            # except (ValueError, OverflowError, ZeroDivisionError):
            #     delta = 0 # Handle errors

        return orders

    def _handle_macarons(self, state: TradingState) -> Tuple[List[Order], int]:
        """Strategy for Magnificent Macarons using observations and conversions."""
        orders = []
        conversions = 0
        product = "MAGNIFICENT_MACARONS"
        position = state.position.get(product, 0)
        max_vol = self.max_volume[product]
        pos_limit = self.position_limit[product]

        # --- Get Market and Observation Data ---
        if product not in state.order_depths: return orders, conversions
        order_depth = state.order_depths[product]
        market_bid, market_ask = self._get_best_bid_ask(order_depth)
        market_mid = self._get_mid_price(product, order_depth)

        # Get Conversion Observation Data
        if not state.observations or not hasattr(state.observations, 'conversionObservations') or product not in state.observations.conversionObservations:
             print("Warning: No conversion observation data for MACARONS.")
             # Fallback or skip if conversion data is crucial
             # return orders, conversions # Option 1: Skip trading
             # Option 2: Fallback to simple MM (as below)
             if market_bid is not None and market_ask is not None:
                 buy_vol = self._compute_order_volume(product, "BUY", position, max_vol // 2)
                 if buy_vol > 0: orders.append(Order(product, market_ask - 1, buy_vol))
                 sell_vol = self._compute_order_volume(product, "SELL", position, max_vol // 2)
                 if sell_vol > 0: orders.append(Order(product, market_bid + 1, -sell_vol))
             return orders, conversions # Return after fallback MM

        obs: ConversionObservation = state.observations.conversionObservations[product]

        # Get Sunlight Data from plainValueObservations
        sunlight = None
        if hasattr(state.observations, 'plainValueObservations') and state.observations.plainValueObservations:
            # Assuming the key for sunlight is 'SUNLIGHT' or similar - adjust if needed
            sunlight_key = 'SUNLIGHT' # Or 'sunlightIndex' ? Check documentation/logs
            if sunlight_key in state.observations.plainValueObservations:
                 sunlight = state.observations.plainValueObservations[sunlight_key]
                 self.last_sunlight = sunlight # Store for potential future use
            else:
                 print(f"Warning: '{sunlight_key}' not found in plainValueObservations.")
        else:
             print("Warning: No plainValueObservations found in state.")

        if sunlight is None:
             print("Warning: Sunlight data unavailable, cannot apply CSI logic.")
             # Decide how to proceed without sunlight data - maybe skip CSI part?
             # For now, we'll proceed but the CSI logic won't trigger if sunlight is None

        # Calculate conversion prices
        cost_to_buy_via_conversion = obs.askPrice + obs.transportFees + obs.importTariff
        revenue_to_sell_via_conversion = obs.bidPrice - obs.transportFees - obs.exportTariff

        # --- Strategy Logic ---
        # 1. CSI Hint: If sunlight < CSI, expect prices to rise. Go long?
        # 2. Conversion Arb: If market price deviates significantly from conversion price.
        # 3. Factor in storage cost for long positions.

        # Define a 'fair value' - could be based on conversion, market, or blend
        fair_value_low = revenue_to_sell_via_conversion
        fair_value_high = cost_to_buy_via_conversion

        # Adjust fair value based on sunlight? Only if sunlight data is available
        if sunlight is not None and sunlight < self.macaron_csi_threshold:
             # If low sunlight, expect prices higher, adjust fair value up?
             fair_value_mid_est = (fair_value_low + fair_value_high) / 2 * 1.01 # Example: 1% higher
        else:
             # Default fair value if sunlight is high or unavailable
             fair_value_mid_est = (fair_value_low + fair_value_high) / 2

        # Adjust for storage cost when considering holding long positions
        storage_cost_adjustment = self.macaron_storage_cost_per_unit_per_tick * 50 # Estimated hold time ticks?

        # --- Trading Decisions ---
        conversions_needed = 0

        # Opportunity 1: Market price is very low (buy from market, potentially sell via conversion later)
        if market_ask is not None and market_ask < fair_value_low - storage_cost_adjustment:
            buy_volume = self._compute_order_volume(product, "BUY", position, max_vol)
            liq = -order_depth.sell_orders.get(market_ask, 0) if market_ask in order_depth.sell_orders else 0
            final_volume = min(buy_volume, liq)
            if final_volume > 0:
                print(f"BUY {product} Market: {final_volume} @ {market_ask} (ConvSell: {revenue_to_sell_via_conversion:.1f})")
                orders.append(Order(product, market_ask, final_volume))

        # Opportunity 2: Market price is very high (sell on market, potentially buy via conversion later)
        elif market_bid is not None and market_bid > fair_value_high:
            sell_volume = self._compute_order_volume(product, "SELL", position, max_vol)
            liq = order_depth.buy_orders.get(market_bid, 0) if market_bid in order_depth.buy_orders else 0
            final_volume = min(sell_volume, liq)
            if final_volume > 0:
                print(f"SELL {product} Market: {final_volume} @ {market_bid} (ConvBuy: {cost_to_buy_via_conversion:.1f})")
                orders.append(Order(product, market_bid, -final_volume))

        # Opportunity 3: Buy via conversion, sell on market
        if market_bid is not None and market_bid > cost_to_buy_via_conversion:
             # How much can we convert? Limited by position limit and conversion limit
             conv_buy_potential = min(self.macaron_conversion_limit, pos_limit - position)
             # How much can we sell on market?
             sell_liq = order_depth.buy_orders.get(market_bid, 0) if market_bid in order_depth.buy_orders else 0
             # Volume is limited by conversion potential and market liquidity
             volume = min(conv_buy_potential, sell_liq)
             volume = max(0, int(volume))
             if volume > 0:
                 print(f"CONVERT_BUY + SELL Mkt {product}: {volume} @ {market_bid} (Cost: {cost_to_buy_via_conversion:.1f})")
                 orders.append(Order(product, market_bid, -volume))
                 conversions_needed += volume # Request conversion to cover the sale

        # Opportunity 4: Buy on market, sell via conversion
        if market_ask is not None and market_ask < revenue_to_sell_via_conversion:
             # How much can we sell via conversion? Limited by position limit and conversion limit
             conv_sell_potential = min(self.macaron_conversion_limit, pos_limit + position)
             # How much can we buy on market?
             buy_liq = -order_depth.sell_orders.get(market_ask, 0) if market_ask in order_depth.sell_orders else 0
             # Volume is limited by conversion potential and market liquidity
             volume = min(conv_sell_potential, buy_liq)
             volume = max(0, int(volume))
             if volume > 0:
                 print(f"BUY Mkt + CONVERT_SELL {product}: {volume} @ {market_ask} (Revenue: {revenue_to_sell_via_conversion:.1f})")
                 orders.append(Order(product, market_ask, volume))
                 conversions_needed -= volume # Request conversion (sell)

        # Adjust conversions based on net position change from market orders
        net_market_qty = sum(o.quantity for o in orders if o.symbol == product)
        # If we bought on market, we might need to sell via conversion later (or vice versa)
        # This simple logic just handles immediate arb.

        # Final conversion request: net amount needed for arb + potentially adjust inventory towards 0?
        # Simple: just fulfill arb needs for now.
        final_conversions = conversions_needed

        # Ensure conversion request respects limits
        if final_conversions > 0: # Buying via conversion
             final_conversions = min(final_conversions, self.macaron_conversion_limit, pos_limit - position)
        elif final_conversions < 0: # Selling via conversion
             final_conversions = max(final_conversions, -self.macaron_conversion_limit, -pos_limit - position)

        # Add basic MM orders if no arb/conversion happened?
        if not orders and market_bid is not None and market_ask is not None:
             buy_vol = self._compute_order_volume(product, "BUY", position, max_vol // 2)
             if buy_vol > 0: orders.append(Order(product, market_ask - 1, buy_vol))
             sell_vol = self._compute_order_volume(product, "SELL", position, max_vol // 2)
             if sell_vol > 0: orders.append(Order(product, market_bid + 1, -sell_vol))


        return orders, int(final_conversions)


    # --- Main Run Method ---
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """Main trading logic loop."""
        result: Dict[str, List[Order]] = {}
        conversions = 0
        trader_data = "" # Optional data to persist state

        # Update price history for relevant products
        for product, order_depth in state.order_depths.items():
             mid_price = self._get_mid_price(product, order_depth)
             self._update_price_history(product, mid_price)

        # --- Handle Picnic Baskets (Arbitrage + MM) ---
        # This needs to be handled together due to shared components
        picnic_orders = self._handle_picnic_baskets(state)
        for order in picnic_orders:
            if order.symbol not in result: result[order.symbol] = []
            result[order.symbol].append(order)
        picnic_related_products = {"PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"}

        # --- Handle Volcanic Vouchers (Options Strategy) ---
        voucher_orders = self._handle_volcanic_vouchers(state)
        for order in voucher_orders:
             if order.symbol not in result: result[order.symbol] = []
             result[order.symbol].append(order)
        voucher_products = set(self.voucher_strikes.keys())

        # --- Handle Macarons (Conversion + Market) ---
        macaron_orders, macaron_conversions = self._handle_macarons(state)
        result["MAGNIFICENT_MACARONS"] = macaron_orders
        conversions += macaron_conversions


        # --- Handle Remaining Products ---
        for product, order_depth in state.order_depths.items():
            # Skip products already handled
            if product in picnic_related_products or product in voucher_products or product == "MAGNIFICENT_MACARONS":
                # Ensure key exists even if no *new* orders were added by individual handlers
                if product not in result: result[product] = []
                continue

            position = state.position.get(product, 0)
            orders = []

            if product == "RAINFOREST_RESIN":
                orders = self._handle_rainforest_resin(order_depth, position)
            elif product == "KELP":
                orders = self._handle_kelp(order_depth, position)
            elif product == "SQUID_INK":
                orders = self._handle_squid_ink(order_depth, position)
            elif product == "VOLCANIC_ROCK":
                 orders = self._handle_volcanic_rock(order_depth, position)
            # Add handlers for any other products if necessary
            # else:
            #      print(f"Warning: No specific handler for product {product}. Skipping.")

            result[product] = orders

        # --- Post-processing / Trader Data ---
        # Example: Persist state if needed
        # trader_data = json.dumps({"last_sunlight": self.last_sunlight})

        # Clean up result dictionary (remove keys with empty order lists)
        # final_result = {k: v for k, v in result.items() if v}

        return result, conversions, trader_data
