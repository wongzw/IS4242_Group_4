import streamlit as st
import plotly.graph_objects as go

# Define the data for each graph
dropdown_data = {
    "Accuracy of models without MediaPipe": {
        "KNN (Train-Test)": 0.09,
        "KNN (Cross Validation)": 0.11,
        "SVM": 0.16,
        "CNN": 0.39
    },
    "Accuracy of models with MediaPipe": {
        "KNN (Train-Test)": 0.03,
        "KNN (Cross Validation)": 0.51,
        "SVM": 0.86,
        "CNN": 0.76
    },
    "AUC of models without MediaPipe": {
        "KNN (Train-Test)": 0.52,
        "KNN (Cross Validation)": 0.93,
        "SVM": 0.52,
        "CNN": 0.92
    },
    "AUC of models with MediaPipe": {
        "KNN (Train-Test)": 0.49,
        "KNN (Cross Validation)": 0.82,
        "SVM": 0.99,
        "CNN": 0.98
    },
    "F1 Score (Weighted Average) of models without MediaPipe": {
        "KNN (Train-Test)": 0.09,
        "KNN (Cross Validation)": 0.1,
        "SVM": 0.16,
        "CNN": 0.39
    },
    "F1 Score (Weighted Average) of models with MediaPipe": {
        "KNN (Train-Test)": 0.03,
        "KNN (Cross Validation)": 0.51,
        "SVM": 0.86,
        "CNN": 0.73
    },
}

# Dropdown to select the graph to display
selected_graph = st.selectbox("Select a graph", list(dropdown_data.keys()))

# Retrieve the data for the selected graph
selected_data = dropdown_data[selected_graph]

# Create a bar chart and set a randomise colour to the graph bars
fig = go.Figure()
for i, (model, value) in enumerate(selected_data.items()):
    # Select a random color
    color = f"hsl({int(360 * i / len(selected_data))}, 80%, 50%)"
    fig.add_trace(go.Bar(x=[model], y=[value], marker_color=color,
                         hovertemplate=f"{model}: {value}<extra></extra>")) # the ending string removes redundant extra values

# Update the chart layout
fig.update_layout(
    title=f"{selected_graph}",
    xaxis_title="Model",
    yaxis_title="Performance",
    font=dict(
        family="Arial",
        size=14,
        color="#7f7f7f"
    ),
    showlegend = False # Hide the legend
)

# Display the chart
st.plotly_chart(fig)