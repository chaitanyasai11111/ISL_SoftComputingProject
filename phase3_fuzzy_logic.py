import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- PHASE 3: FUZZY LOGIC SUPERVISOR MODULE ---

class FuzzySupervisor:
    def __init__(self):
        print("Initializing Fuzzy Rule Blocks...")
        self.uv_simulator = self._build_uv_rules()
        self.fist_simulator = self._build_fist_rules()

    def _build_uv_rules(self):
        """Rule Block 1: For open-finger ambiguities (e.g., U vs V)"""
        defect_depth = ctrl.Antecedent(np.arange(0, 1.05, 0.05), 'defect_depth')
        v_confidence = ctrl.Consequent(np.arange(0, 101, 1), 'v_confidence')
        
        defect_depth['shallow'] = fuzz.trimf(defect_depth.universe, [0, 0, 0.3])
        defect_depth['medium'] = fuzz.trimf(defect_depth.universe, [0.15, 0.4, 0.7])
        defect_depth['deep'] = fuzz.trimf(defect_depth.universe, [0.5, 1.0, 1.0])
        
        v_confidence['low'] = fuzz.trimf(v_confidence.universe, [0, 0, 40])
        v_confidence['medium'] = fuzz.trimf(v_confidence.universe, [20, 50, 80])
        v_confidence['high'] = fuzz.trimf(v_confidence.universe, [60, 100, 100])
        
        rule1 = ctrl.Rule(defect_depth['shallow'], v_confidence['low'])
        rule2 = ctrl.Rule(defect_depth['medium'], v_confidence['medium'])
        rule3 = ctrl.Rule(defect_depth['deep'], v_confidence['high'])
        
        fis_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        return ctrl.ControlSystemSimulation(fis_ctrl)

    def _build_fist_rules(self):
        """Rule Block 2: For closed-fist ambiguities (e.g., A vs S vs E)"""
        # Using the first Hu Moment (spread/mass) to determine tightness of the fist
        hu_spread = ctrl.Antecedent(np.arange(0, 5.0, 0.1), 'hu_spread')
        tightness = ctrl.Consequent(np.arange(0, 101, 1), 'tightness')
        
        hu_spread['compact'] = fuzz.trimf(hu_spread.universe, [0, 0, 2.0])
        hu_spread['loose'] = fuzz.trimf(hu_spread.universe, [1.5, 5.0, 5.0])
        
        tightness['high'] = fuzz.trimf(tightness.universe, [50, 100, 100]) # Likely 'S' (tight fist)
        tightness['low'] = fuzz.trimf(tightness.universe, [0, 0, 60])      # Likely 'E' or 'A' (looser)
        
        rule1 = ctrl.Rule(hu_spread['compact'], tightness['high'])
        rule2 = ctrl.Rule(hu_spread['loose'], tightness['low'])
        
        fis_ctrl = ctrl.ControlSystem([rule1, rule2])
        return ctrl.ControlSystemSimulation(fis_ctrl)

    def evaluate(self, top_two_classes, features):
        """
        Takes the top 2 confused classes from the ANN and the 9 physical features.
        Returns the resolved class string, or None if it cannot decide.
        """
        # Feature Mapping (Based on Phase 1 extraction)
        hu_moment_0 = abs(features[0]) 
        defect_count = features[7]
        defect_depth = min(features[8], 1.0) # Ensure it caps at 1.0 for the FIS

        # --- CLUSTER 1: U vs V ---
        if 'U' in top_two_classes and 'V' in top_two_classes:
            self.uv_simulator.input['defect_depth'] = defect_depth
            self.uv_simulator.compute()
            score = self.uv_simulator.output['v_confidence']
            
            if score >= 60: return 'V'
            elif score <= 40: return 'U'
            
        # --- CLUSTER 2: A vs S ---
        # Both are closed fists, so defect_count is usually 0.
        elif 'A' in top_two_classes and 'S' in top_two_classes and defect_count == 0:
            # Clamp the Hu moment to our universe of discourse (0 to 5)
            self.fist_simulator.input['hu_spread'] = min(hu_moment_0, 4.9)
            self.fist_simulator.compute()
            score = self.fist_simulator.output['tightness']
            
            if score >= 60: return 'S' # S is a tighter fist than A
            elif score <= 40: return 'A'

        # If it doesn't fall into a known confusion cluster, defer to Neural Network
        return None 

def generate_report_visuals():
    """Generates the Fuzzy Membership Function plots for the project report."""
    print("Generating Fuzzy Membership Graphs for the Project Report...")
    
    # Re-create variables just for plotting
    defect_depth = ctrl.Antecedent(np.arange(0, 1.05, 0.05), 'defect_depth')
    v_confidence = ctrl.Consequent(np.arange(0, 101, 1), 'v_confidence')
    
    defect_depth['shallow'] = fuzz.trimf(defect_depth.universe, [0, 0, 0.3])
    defect_depth['medium'] = fuzz.trimf(defect_depth.universe, [0.15, 0.4, 0.7])
    defect_depth['deep'] = fuzz.trimf(defect_depth.universe, [0.5, 1.0, 1.0])
    
    v_confidence['low'] = fuzz.trimf(v_confidence.universe, [0, 0, 40])
    v_confidence['medium'] = fuzz.trimf(v_confidence.universe, [20, 50, 80])
    v_confidence['high'] = fuzz.trimf(v_confidence.universe, [60, 100, 100])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(defect_depth.universe, fuzz.trimf(defect_depth.universe, [0, 0, 0.3]), 'b', linewidth=2, label='Shallow')
    ax1.plot(defect_depth.universe, fuzz.trimf(defect_depth.universe, [0.15, 0.4, 0.7]), 'g', linewidth=2, label='Medium')
    ax1.plot(defect_depth.universe, fuzz.trimf(defect_depth.universe, [0.5, 1.0, 1.0]), 'r', linewidth=2, label='Deep')
    ax1.set_title('Antecedent: Convexity Defect Depth')
    ax1.set_xlabel('Normalized Depth')
    ax1.set_ylabel('Membership Degree')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(v_confidence.universe, fuzz.trimf(v_confidence.universe, [0, 0, 40]), 'b', linewidth=2, label='Low')
    ax2.plot(v_confidence.universe, fuzz.trimf(v_confidence.universe, [20, 50, 80]), 'g', linewidth=2, label='Medium')
    ax2.plot(v_confidence.universe, fuzz.trimf(v_confidence.universe, [60, 100, 100]), 'r', linewidth=2, label='High')
    ax2.set_title('Consequent: Confidence in V-Shape')
    ax2.set_xlabel('Probability Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fuzzy_membership_functions.png')
    print("Saved 'fuzzy_membership_functions.png'.")

if __name__ == "__main__":
    # If run directly, just generate the graphs for the report.
    generate_report_visuals()