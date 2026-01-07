import math
import random
import numpy as np
import copy

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # Dictionary: {'t': time, 'occ': occupancy, 'daily_revenue': rev}
        self.parent = parent
        self.action = action  # Action taken to reach this node (e.g., price_change)
        self.children = []
        self.visits = 0
        self.value = 0.0  # Total accumulated reward (Revenue)

    def is_fully_expanded(self, valid_actions):
        return len(self.children) == len(valid_actions)

    def best_child(self, exploration_weight=1.41):
        # Upper Confidence Bound (UCT)
        choices_weights = [
            (child.value / child.visits) + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

class LiteSimulator:
    """
    A lightweight 'Mental Model' of the parking lot environment.
    Used by MCTS to hallucinate future outcomes.
    """
    def __init__(self, forecast_df, capacity=180, base_price=1.5):
        self.forecast_df = forecast_df
        self.capacity = capacity
        self.base_price = base_price
        
        # Internal Beliefs (The AI's model of the world)
        # We start with the 'known' tuned parameter k=2.0
        self.belief_elasticity_k = 2.0 
        self.belief_traffic_impact = 0.3

    def _get_forecast_at(self, t_minute):
        """Retrieves forecasted environment state at time t."""
        # Convert t_minute (sim time) to DataFrame index
        # Assuming forecast_df is indexed by minute or we use iloc
        # Simple lookup: t_minute is index in forecast array if aligned
        # For robustness, we'll assume forecast_df is passed as a list/dict or handled 
        # by the caller to align properly. 
        # Let's assume forecast_df is just the rows starting from current time.
        idx = int(t_minute / 5) # 5 min intervals
        if idx >= len(self.forecast_df):
            return self.forecast_df.iloc[-1]
        return self.forecast_df.iloc[idx]

    def predict_step(self, state, action):
        """
        Simulates one step forward (5 mins).
        Args:
            state: {'t': mins_from_now, 'occ': current_occ, 'price': current_price}
            action: price_change (float)
        Returns:
            next_state, reward (revenue)
        """
        t = state['t']
        current_occ = state['occ']
        current_price = state['price'] + action
        
        # Hard constraints on price
        current_price = max(0.0, current_price) # No negative price

        # 1. Get Environmental Forecast
        # We assume forecast_df is sliced so index 0 is NOW
        # t is in minutes relative to start of planning
        row = self._get_forecast_at(t)
        
        # 2. Base Demand (Uncapped)
        base_occ = row.get('occupancy_pred', 100) # Default if missing
        avg_temp = row.get('avg_temp', 20.0)
        
        # 3. Apply Elasticity Belief
        # Re-implementing the sigmoid logic locally (Mental Model)
        weather_score = min(1.0, abs(avg_temp - 20.0) / 10.0)
        
        # Normalized Price Ratio
        if self.base_price > 0:
            ratio = current_price / self.base_price
        else:
            ratio = 1.0
            
        # Sigmoid Function
        effective_ratio = ratio / (1.0 + 0.3 * weather_score)
        val = self.belief_elasticity_k * (effective_ratio - 1.0)
        val = max(-100, min(100, val))
        price_factor = 2.0 / (1.0 + np.exp(val))
        
        if effective_ratio < 1.05:
             price_factor *= (1.0 + 0.5 * weather_score)
             
        # Traffic Factor
        traffic_factor = 1.0 # Simplify for lite model or implement if critical
        
        # 4. Resulting Occupancy
        target_occ = base_occ * price_factor # * traffic_factor
        final_occ = max(0.0, min(target_occ, self.capacity))
        
        # 5. Reward = Revenue
        # Revenue for this 5-min step
        step_revenue = final_occ * current_price
        
        next_state = {
            't': t + 5, # Advance 5 mins
            'occ': final_occ,
            'price': current_price
        }
        
        return next_state, step_revenue

class MCTSAgent:
    def __init__(self, forecast_df, capacity=180, base_price=1.5, planning_horizon=60, n_simulations=50):
        self.simulator = LiteSimulator(forecast_df, capacity, base_price)
        self.horizon = planning_horizon # lookahead in minutes
        self.n_sims = n_simulations
        self.actions = [0.0, 0.10, -0.10, 0.25, -0.25] # Discrete price moves

    def get_action(self, current_state):
        """
        current_state: {'occ': ..., 'price': ...}
        """
        # Root Node
        root_state = {'t': current_state['t'], 'occ': current_state['occ'], 'price': current_state['price']}
        root = MCTSNode(root_state)

        for _ in range(self.n_sims):
            node = root
            
            # 1. Selection
            while node.is_fully_expanded(self.actions) and node.children:
                node = node.best_child()

            # 2. Expansion
            if not node.is_fully_expanded(self.actions) and node.state['t'] < self.horizon:
                # Pick an untried action
                tried_actions = [child.action for child in node.children]
                untried = [a for a in self.actions if a not in tried_actions]
                action = random.choice(untried)
                
                # Create child
                next_state, reward = self.simulator.predict_step(node.state, action)
                new_node = MCTSNode(next_state, parent=node, action=action)
                node.children.append(new_node)
                node = new_node
                
            # 3. Simulation (Rollout)
            # Random walk to end of horizon
            curr = node.state
            accum_reward = 0.0
            depth = 0
            
            while curr['t'] < self.horizon:
                rand_action = random.choice(self.actions)
                curr, r = self.simulator.predict_step(curr, rand_action)
                accum_reward += r
                depth += 1
            
            # 4. Backpropagation
            # Propagate value up
            while node is not None:
                node.visits += 1
                node.value += accum_reward
                node = node.parent
                
        # Select best action (most visited usually robust)
        if not root.children:
            return 0.0
            
        return max(root.children, key=lambda c: c.visits).action
