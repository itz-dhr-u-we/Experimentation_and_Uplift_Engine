import gradio as gr
import matplotlib.pyplot as plt
from src.causal_model import MobilityCausalAnalyzer
from src.simulator import generate_experiment_data
import plotly.graph_objects as go 

generate_experiment_data() 
analyzer = MobilityCausalAnalyzer() 

#  links inputs to outputs
def evaluate_cohort(distance, wait_time, device, history, surge):
    is_peak = 1 if device == "Peak Hour" else 0
    
    # Run the causal inference prediction
    ctrl_p, treat_p, lift, churn = analyzer.predict_marketplace_impact(
        distance, wait_time, is_peak, history, surge
    )

    # Logic for the "Strategic Decision" message
    recommendation = (
        " SHIP FEATURE: High positive uplift. Roll out surge cap."
        if lift > 0.02 else " WITHHOLD: Neutral/Negative impact."
    )

    # interactive Chart
    fig = go.Figure(data=[
        go.Bar(name='Control', x=['Completion Rate'], y=[ctrl_p * 100], 
               marker_color='#6c757d'),
        go.Bar(name='Treatment', x=['Completion Rate'], y=[treat_p * 100], 
               marker_color='#0d6efd' if lift > 0 else '#dc3545')
    ])
    
    fig.update_layout(
        barmode='group',
        title="Causal Effect of Surge Pricing Intervention",
        yaxis=dict(title="Completion Rate (%)", range=[0, 100]),
        height=300
    )

    return (f"{ctrl_p*100:.1f}%", f"{lift*100:+.2f}%", recommendation, fig)

#  Gradio UI Layout
demo = gr.Interface(
    fn=evaluate_cohort,
    inputs=[
        gr.Slider(1, 25, value=5, label="Trip Distance (km)"),
        gr.Slider(1, 20, value=7, label="Driver Wait Time (min)"),
        gr.Radio(["Normal", "Peak Hour"], label="Time of Day"),
        gr.Slider(0, 50, value=10, label="Rider Lifetime History"),
        gr.Slider(1.0, 3.5, value=1.5, step=0.1, label="Current Surge Multiplier"),
    ],
    outputs=[
        gr.Textbox(label="Baseline Completion Rate"),
        gr.Textbox(label="Estimated Uplift"),
        gr.Markdown(label="Business Decision Rule"),
        gr.Plot(label="Marketplace Impact Chart"),
    ],
    title="Surge Pricing Causal Analyzer",
    description="Optimizing surge pricing strategies to balance rider conversion vs. churn."
)

if __name__ == "__main__":
    demo.launch(share=True)