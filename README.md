# Surge Pricing Causal Analyzer

A production-grade machine learning and causal inference pipeline designed to optimize ride-sharing dynamic surge pricing, balancing short-term rider conversion rates against long-term customer churn risk.

# About the Project
In multi-sided mobility marketplaces (such as Uber, Lyft, or Ola), dynamic surge pricing helps balance hyper-local driver supply and rider demand during peak hours. However, excessive price hikes often alienate users, leading to permanent app uninstalls (churn). 

This project implements a Heterogeneous Treatment Effect (HTE) uplift model using Gradient Boosting and counterfactual analysis. Instead of treating all users uniformly, the system predicts how individual customer segments react to an AI-driven price-capping policy versus legacy pricing, ensuring dynamic surge caps are deployed only when they yield positive net conversion lift.


# How to Run Locally

1. Clone or download the repository:
   git clone [https://github.com/itz-dhr-u-we/Experimentation_and_Uplift_Engine.git](https://github.com/itz-dhr-u-we/Experimentation_and_Uplift_Engine.git)
   
2. Install the required dependencies:
    pip install -r requirements.txt

3. Run the application:
    python app.py

4. Open the local web address provided in your terminal