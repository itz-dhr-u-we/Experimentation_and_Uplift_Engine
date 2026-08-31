import os
import numpy as np 
import pandas as pd 

def generate_experiment_data(n_sample=3000,save_path="data/raw_experiments.csv"):
    """Simulates a Ride-Sharing Surge Pricing A/B experiment dataset with heterogeneous treatment effects (HTE)."""
    np.random.seed(42)

    user_id = [f"RIDER_{i:04d}" for i in range(n_sample)]#acting as primary key in data schema
    trip_distance_km = np.round(
        np.random.uniform(low=1.5, high=35.0, size=n_sample), 2
    )
    estimated_wait_time_min = np.random.poisson(lam=7, size=n_sample)
    is_peak_hour = np.random.choice([0, 1], size=n_sample, p=[0.60, 0.40])
    historical_rides = np.random.geometric(p=0.08, size=n_sample)#more new users(< 10 rides) then power users(> 100 rides)

    #control,treatment - A/B Testing
    treatment = np.random.choice([0, 1], size=n_sample, p=[0.5, 0.5])
    available_drivers_nearby = np.random.poisson(lam=5, size=n_sample)

    supply_pressure_factor = np.where(is_peak_hour == 1, 3.5 / (available_drivers_nearby + 1), 1.0)
    base_surge = 1.0 + (0.5 * supply_pressure_factor) + (0.02 * trip_distance_km)
    #base_surge = 1.0 + (0.8 * is_peak_hour) + (0.02 * trip_distance_km)
    surge_discount = treatment * (0.25 * (historical_rides > 15) - 0.15 * is_peak_hour)
    final_surge_multiplier = np.round(
        np.clip(base_surge - surge_discount, 1.0, 3.5), 2
    )

    base_completion_prob = (
        0.85 - (0.18 * (final_surge_multiplier - 1.0)) - (0.02 * estimated_wait_time_min)
    )
    loyalty_boost = 0.005 * historical_rides
    final_completion_prob = np.clip(
        base_completion_prob + loyalty_boost, 0.05, 0.98
    )

    ride_completed = np.random.binomial(1, final_completion_prob)

    churn_prob = np.where(
        (final_surge_multiplier > 2.2) & (ride_completed == 0), 0.35, 0.03
    )
    rider_churned = np.random.binomial(1, churn_prob)

    df = pd.DataFrame({
        "user_id": user_id,
        "trip_distance_km": trip_distance_km,
        "estimated_wait_time_min": estimated_wait_time_min,
        "is_peak_hour": is_peak_hour,
        "historical_rides": historical_rides,
        "treatment": treatment,
        "surge_multiplier": final_surge_multiplier,
        "ride_completed": ride_completed,
        "rider_churned": rider_churned,
    })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(
        f"[SUCCESS] Mobility experimentation logs generated and saved to:"
        f" {save_path}"
    )
    return df


if __name__ == "__main__":
  generate_experiment_data()
    