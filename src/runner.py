import yaml
import os
import argparse
import sys

# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "digital_twin_sim"))

from policies import make_policy

# Import Refactored Reporting Module
from reporting import (
    calculate_metrics, 
    save_csv_results, 
    save_simple_plot, 
    save_comparison_plot_simple
)

try:
    from digital_twin_sim.twin_offline_sim import OfflineDigitalTwinSimulator
except ImportError:
    from digital_twin_sim.twin_offline_sim import OfflineDigitalTwinSimulator

os.makedirs("reports", exist_ok=True)

# ---------------------------------------------------------
# TWIN ANALYSIS HELPER
# ---------------------------------------------------------
def run_twin_analysis(cfg, args, test_policy_type, forecaster_type, label_test, scenarios=None):
    if scenarios is None:
        scenarios = {"Base": 1.0, "2x_Demand": 2.0, "3x_Demand": 3.0}

    lot_capacity = cfg["lots"][0]["capacity"]
    
    for sc_name, sc_mult in scenarios.items():
        print(f"\n>>> Running Scenario: {sc_name} (Multiplier: {sc_mult}x)")
        try: sys.stdout.flush() 
        except: pass
        
        cfg["demand_multiplier"] = sc_mult
        
        # 1. Benchmark: STATIC
        print(f"    [{sc_name}] Running Benchmark: STATIC (Env: {forecaster_type})...")
        sim_s = OfflineDigitalTwinSimulator(cfg)
        log_s, df_f = sim_s.run(make_policy("static"), forecaster_type=forecaster_type)
        rev_s, price_s, occ_s, rej_s, served_s = calculate_metrics(log_s, capacity=lot_capacity)
        
        # 2. Test Policy
        print(f"    [{sc_name}] Running Test: {label_test.upper()}...")
        sim_d = OfflineDigitalTwinSimulator(cfg)
        
        # Configure Policy
        policy = make_policy(test_policy_type)
        
        # FIX: Force YAML config injection into Policy object
        if hasattr(policy, "target"): 
            policy.target = cfg.get("target", 0.8) # Load from YAML if available
        if hasattr(policy, "k"):
            policy.k = cfg.get("k", 1.0)
        if hasattr(policy, "p_min"):
            policy.p_min = cfg.get("p_min", 0.0)
        if hasattr(policy, "p_max"):
            policy.p_max = cfg.get("p_max", 5.0)

        # CLI overrides
        if hasattr(policy, "target") and args.target is not None: policy.target = args.target
        if hasattr(policy, "k") and args.k is not None: policy.k = args.k
        
        log_d, _ = sim_d.run(policy, forecaster_type=forecaster_type)
        rev_d, price_d, occ_d, rej_d, served_d = calculate_metrics(log_d, capacity=lot_capacity)
        
        # Calculate Delta
        if served_s > 0:
             delta_demand_pct = ((served_d - served_s) / served_s) * 100.0
        else:
             delta_demand_pct = 0.0

        print("\n" + "=" * 94)
        print(f"SCENARIO: {sc_name:<10} | {'METRIC':<18} | {'STATIC':<10} | {label_test.upper():<12} | {'DELTA':<8}")
        print("-" * 94)
        print(f"{' ':24} | {'Revenue (€)':<18} | {rev_s:10.2f} | {rev_d:10.2f} | {rev_d - rev_s:+8.2f}")
        print(f"{' ':24} | {'Avg Price (€)':<18} | {price_s:10.2f} | {price_d:10.2f} | {price_d - price_s:+8.2f}")
        print(f"{' ':24} | {'Avg Occ (%)':<18} | {occ_s*100:10.1f} | {occ_d*100:10.1f} | {(occ_d - occ_s)*100:+8.1f}")
        print(f"{' ':24} | {'Rejection (%)':<18} | {rej_s:10.1f} | {rej_d:10.1f} | {rej_d - rej_s:+8.1f}") 
        print(f"{' ':24} | {'Demand Delta(%)':<18} | {'---':^10} | {delta_demand_pct:+10.1f} | {' ':>8}")
        print("=" * 94 + "\n")
        
        suffix = f"_{label_test}_{sc_name}"
        save_simple_plot(log_d, df_f, filename=f"simple_plot{suffix}.png", label=label_test)
        save_csv_results(log_s, log_d, df_f, filename=f"comparison_data{suffix}.csv",
                         label_1="Static", label_2=label_test)

# ---------------------------------------------------------
# MAIN EXECUTION ENTRY POINT
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=int, choices=[1, 2, 3], default=None,
        help="1=Classic, 2=AI, 3=Battle(Classic vs AI)"
    )
    
    # --- ARGUMENTS FOR SCENARIO RUNNER ---
    parser.add_argument("--target", type=float, default=None, help="Target Occupancy (0.0 - 1.0)")
    parser.add_argument("--k", type=float, default=None, help="Controller Gain K")
    parser.add_argument("--p_min", type=float, default=None, help="Minimum Price")
    parser.add_argument("--silent", action="store_true", help="Output only JSON for automation")
    # -------------------------------------

    args = parser.parse_args()

    # Load Configuration
    config_path = "config/parking_P023.yaml"
    if not os.path.exists(config_path):
        config_path = "../config/parking_P023.yaml"
    
    try:
        cfg = yaml.safe_load(open(config_path))
    except Exception as e:
        if not args.silent:
            print(f"[Error] Failed to load config: {e}")
        return

    # Determine Execution Mode
    mode = args.mode
    
    if mode is None:
        if args.silent:
            mode = 1
        else:
            print("\n================ Execution Mode ================")
            print("1 - Classic Analysis (Static vs Integral Control)")
            print("2 - AI Analysis (Static vs MCTS/Chronos)")
            print("3 - Battle Mode (Classic vs AI)")
            try:
                choice = input("\nSelect (1, 2 or 3): ").strip()
                if choice not in ("1", "2", "3"):
                    mode = 1
                else:
                    mode = int(choice)
            except:
                mode = 1

    # Apply overrides just to the dict (but we must inject them manually later)
    if args.target is not None: cfg["policy"]["target_occupancy"] = args.target
    if args.k is not None: cfg["policy"]["k"] = args.k
    if args.p_min is not None: cfg["policy"]["p_min"] = args.p_min

    # ---------------------------------------------------------
    # MODE 1 — Classic Analysis
    # ---------------------------------------------------------
    if mode == 1:
        if not args.silent:
            print("\n[MODE 1] Classic Analysis: Static vs Integral Control (Balanced)\n")
        
        run_twin_analysis(cfg, args, 
                          test_policy_type="dynpricing", 
                          forecaster_type="statistical", 
                          label_test="Classic")

    # ---------------------------------------------------------
    # MODE 2 — AI Analysis
    # ---------------------------------------------------------
    elif mode == 2:
        if not args.silent:
            print("\n[MODE 2] AI Analysis: Static vs MCTS Agent (Chronos)\n")
        
        run_twin_analysis(cfg, args, 
                          test_policy_type="mcts", 
                          forecaster_type="chronos", 
                          label_test="AI")

    # ---------------------------------------------------------
    # MODE 3 — Battle Mode (Classic vs AI)
    # ---------------------------------------------------------
    elif mode == 3:
        if not args.silent:
            print("\n[MODE 3] Battle Mode: Classic (Statistical) vs AI (Chronos + MCTS)\n")

        lot_capacity = cfg["lots"][0]["capacity"]
        scenarios = {"3x_Demand": 3.0}

        # --- LOAD YAML PARAMS ---
        # Read from YAML once to use in both contenders
        # NOTE: This fixes the issue where defaults were ignoring the YAML file
        yaml_target = cfg.get("target", 0.8)
        yaml_k      = cfg.get("k", 1.0)
        yaml_pmin   = cfg.get("p_min", 0.0)
        yaml_pmax   = cfg.get("p_max", 5.0)

        # Allow CLI override
        if args.target is not None: yaml_target = args.target
        if args.k is not None:      yaml_k = args.k
        if args.p_min is not None:  yaml_pmin = args.p_min
        
        for sc_name, sc_mult in scenarios.items():
            print(f"\n>>> Running Scenario: {sc_name} (Multiplier: {sc_mult}x)")
            try: sys.stdout.flush() 
            except: pass
            
            cfg["demand_multiplier"] = sc_mult
            
            # --- RUN 1: CLASSIC (Benchmark) ---
            print(f"    [{sc_name}] Running Contender 1: CLASSIC (Using Chronos Env for Fairness)...")
            sim_c = OfflineDigitalTwinSimulator(cfg)
            pol_c = make_policy("dynpricing")
            
            # INJECT CONFIG
            pol_c.target = yaml_target
            pol_c.k = yaml_k
            pol_c.p_min = yaml_pmin
            pol_c.p_max = yaml_pmax
            
            # CRITICAL: Use 'chronos' for environment generation so comparisons are fair
            log_c, df_f_c = sim_c.run(pol_c, forecaster_type="chronos")
            rev_c, price_c, occ_c, rej_c, served_c = calculate_metrics(log_c, capacity=lot_capacity)
            
            # --- RUN 2: AI (Challenger) ---
            print(f"    [{sc_name}] Running Contender 2: AI (Chronos + MCTS)...")
            sim_ai = OfflineDigitalTwinSimulator(cfg)
            pol_ai = make_policy("mcts")
            
            # INJECT CONFIG INTO AI AGENT
            pol_ai.p_min = yaml_pmin
            pol_ai.p_max = yaml_pmax
            
            log_ai, df_f_ai = sim_ai.run(pol_ai, forecaster_type="chronos")
            rev_ai, price_ai, occ_ai, rej_ai, served_ai = calculate_metrics(log_ai, capacity=lot_capacity)
            
             # Calculate Delta
            if served_c > 0:
                 delta_demand_pct = ((served_ai - served_c) / served_c) * 100.0
            else:
                 delta_demand_pct = 0.0

            # Output
            print("\n" + "=" * 94)
            print(f"SCENARIO: {sc_name:<10} | {'METRIC':<18} | {'CLASSIC':<10} | {'AI':<12} | {'DELTA':<8}")
            print("-" * 94)
            print(f"{' ':24} | {'Revenue (€)':<18} | {rev_c:10.2f} | {rev_ai:10.2f} | {rev_ai - rev_c:+8.2f}")
            print(f"{' ':24} | {'Avg Price (€)':<18} | {price_c:10.2f} | {price_ai:10.2f} | {price_ai - price_c:+8.2f}")
            print(f"{' ':24} | {'Avg Occ (%)':<18} | {occ_c*100:10.1f} | {occ_ai*100:10.1f} | {(occ_ai - occ_c)*100:+8.1f}")
            print(f"{' ':24} | {'Rejection (%)':<18} | {rej_c:10.1f} | {rej_ai:10.1f} | {rej_ai - rej_c:+8.1f}") 
            print(f"{' ':24} | {'Demand Delta(%)':<18} | {'---':^10} | {delta_demand_pct:+10.1f} | {' ':>8}")
            print("=" * 94 + "\n")
            
            suffix = f"_Battle_{sc_name}"
            # Uses simplified comparison plot (Forecast vs Classic vs AI)
            save_comparison_plot_simple(log_c, log_ai, df_f_ai, filename=f"comparison_simple{suffix}.png", 
                             label_1="Classic", label_2="AI")
            
            save_csv_results(log_c, log_ai, df_f_ai, filename=f"comparison_data{suffix}.csv",
                             label_1="Classic", label_2="AI")


if __name__ == "__main__":
    main()