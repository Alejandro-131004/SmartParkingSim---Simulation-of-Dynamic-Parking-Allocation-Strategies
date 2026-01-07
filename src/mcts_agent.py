import math
import random
import numpy as np
import copy

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # {'t': time, 'occ': occupancy, 'price': price}
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
            (child.value / max(1, child.visits)) + exploration_weight * math.sqrt(math.log(max(1, self.visits)) / max(1, child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

class LiteSimulator:
    """
    A lightweight 'Mental Model' of the parking lot environment.
    MUST MATCH twin_offline_sim.py LOGIC EXACTLY.
    """
    def __init__(self, forecast_df, capacity=180, base_price=1.5):
        self.forecast_df = forecast_df
        self.capacity = capacity
        self.base_price = base_price
        
        # Internal Beliefs (Aligned with Twin Defaults)
        self.belief_elasticity_k = 5.0 # Match 'k' in sigmoid of Twin
        self.belief_traffic_impact = 0.3

    def _get_forecast_at(self, t_minute):
        idx = int(t_minute / 5) # 5 min intervals
        if idx >= len(self.forecast_df):
            return self.forecast_df.iloc[-1]
        return self.forecast_df.iloc[idx]

    def predict_step(self, state, action):
        """
        Simulates one step forward (5 mins).
        """
        t = state['t']
        current_occ = state['occ']
        current_price = state['price'] + action
        
        # Hard constraints on price handled by caller or clamped here
        current_price = max(0.0, current_price) 

        # 1. Get Environmental Forecast
        row = self._get_forecast_at(t)
        
        # 2. Base Demand
        # Use 'occupancy_pred' as the base potential demand (uncapped)
        # Note: Forecast DF usually has 'occupancy_pred' scaled to vehicle count or ratio?
        # In Twin we treat 'occupancy_pred' as raw demand.
        base_occ = row.get('occupancy_pred', 0) 
        avg_temp = row.get('avg_temp', 20.0)
        
        # 3. Apply Elasticity (Sigmoid)
        # Matches twin_offline_sim.py _calculate_sigmoid_elasticity logic
        weather_score = min(1.0, abs(avg_temp - 20.0) / 10.0)
        
        if self.base_price > 0:
            ratio = current_price / self.base_price
        else:
            ratio = 1.0
            
        effective_ratio = ratio / (1.0 + 0.3 * weather_score)
        val = self.belief_elasticity_k * (effective_ratio - 1.0)
        val = max(-100.0, min(100.0, val))
        
        price_factor = 2.0 / (1.0 + np.exp(val))
        
        if effective_ratio < 1.05:
             price_factor *= (1.0 + 0.5 * weather_score)
             
        # Traffic Factor (Simple belief)
        traffic_factor = 1.0 
        
        # 4. Resulting Occupancy
        target_occ = base_occ * price_factor * traffic_factor
        final_occ = max(0.0, min(target_occ, self.capacity))
        
        # 5. Reward = Revenue (Estimated)
        step_revenue = final_occ * current_price
        
        next_state = {
            't': t + 5, # Advance 5 mins
            'occ': final_occ,
            'price': current_price
        }
        
        return next_state, step_revenue

class MCTSAgent:
    def __init__(self, forecast_df, capacity=180, base_price=1.5, planning_horizon=60, n_simulations=50, p_min=0.0, p_max=50.0):
        self.simulator = LiteSimulator(forecast_df, capacity, base_price)
        self.horizon = planning_horizon
        self.n_sims = n_simulations
        self.p_min = p_min
        self.p_max = p_max
        # Discrete actions: Maintain, Small Nudge, Big Nudge
        self.actions = [0.0, 0.10, -0.10, 0.25, -0.25] 

    def get_action(self, current_state):
        root_state = {'t': current_state['t'], 'occ': current_state['occ'], 'price': current_state['price']}
        root = MCTSNode(root_state)

        for _ in range(self.n_sims):
            node = root
            
            # 1. Selection
            # Find a leaf node that hasn't been fully expanded
            while node.children:
                # Filter actions that are valid (within p_min/p_max)
                valid_actions = self._get_valid_actions(node.state['price'])
                if not node.is_fully_expanded(valid_actions):
                    break
                node = node.best_child()

            # 2. Expansion
            valid_actions = self._get_valid_actions(node.state['price'])
            tried_actions = [child.action for child in node.children]
            untried = [a for a in valid_actions if a not in tried_actions]
            
            if untried and node.state['t'] < self.horizon:
                action = random.choice(untried)
                next_state, reward = self.simulator.predict_step(node.state, action)
                new_node = MCTSNode(next_state, parent=node, action=action)
                node.children.append(new_node)
                node = new_node
                
            # 3. Simulation (Rollout)
            curr = node.state
            accum_reward = 0.0
            depth = 0
            
            # Fast rollout
            while curr['t'] < self.horizon:
                # Random valid walk
                valid_rollout = self._get_valid_actions(curr['price'])
                rand_action = random.choice(valid_rollout)
                curr, r = self.simulator.predict_step(curr, rand_action)
                accum_reward += r
                depth += 1
            
            # 4. Backpropagation
            while node is not None:
                node.visits += 1
                node.value += accum_reward
                node = node.parent
                
        # Select best action
        if not root.children:
            return 0.0
            
        return max(root.children, key=lambda c: c.visits).action

    def _get_valid_actions(self, current_price):
        """Returns list of actions that keep price within bounds."""
        valid = []
        for a in self.actions:
            p = current_price + a
            if self.p_min <= p <= self.p_max:
                valid.append(a)
        # Always allow 0.0 (maintain) to avoid getting stuck
        if not valid: valid = [0.0] 
        return valid