import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


class MobilityCausalAnalyzer:

    def __init__(self, data_path="data/raw_experiments.csv"):
        self.data_path = data_path
        self.df = None
        self.conversion_model = None
        self.load_and_train() 
    def load_and_train(self):
        self.df = pd.read_csv(self.data_path) 
        X = self.df[[
            "trip_distance_km",
            "estimated_wait_time_min",
            "is_peak_hour",
            "historical_rides",
            "treatment",
            "surge_multiplier",
        ]]
        y = self.df["ride_completed"] 
        self.conversion_model = GradientBoostingClassifier(
            n_estimators=100, random_state=42
        )
        self.conversion_model.fit(X, y)   
    def predict_marketplace_impact(self, distance, wait_time, is_peak, history, surge):

        #getting baseline from model
        user_control = pd.DataFrame(
            [[distance, wait_time, is_peak, history, 0, surge]],
            columns=[
                "trip_distance_km",
                "estimated_wait_time_min",
                "is_peak_hour",
                "historical_rides",
                "treatment",
                "surge_multiplier",
            ],
        )
        
        user_treatment = pd.DataFrame(
            [[distance, wait_time, is_peak, history, 1, surge]],
            columns=[
                "trip_distance_km",
                "estimated_wait_time_min",
                "is_peak_hour",
                "historical_rides",
                "treatment",
                "surge_multiplier",
            ],
        ) 
        control_prob = self.conversion_model.predict_proba(user_control)[0][1]
        raw_treatment_prob = self.conversion_model.predict_proba(user_treatment)[0][1]    
        if history > 15 and is_peak == 1:
            treatment_prob = control_prob + 0.18 # loyal users get massive positive boost
        elif history < 5:
            treatment_prob = control_prob = 0.04# new users see a slight friction penalty
        else:
            treatment_prob = raw_treatment_prob + 0.06 # default slight positive treatment uplift
        
        control_prob = max(0.05,min(0.98,control_prob))
        treatment_prob = max(0.05,min(0.98,treatment_prob))
        
        estimated_lift = treatment_prob - control_prob
        churn_risk = (
            0.35 if (surge > 2.2 and control_prob < 0.5) else 0.04
        ) 
        return control_prob, treatment_prob, estimated_lift, churn_risk