import streamlit as st
import numpy as np
import torch

st.title("SpMV-CNN Model: Runtime & Energy Consumption Estimator")
st.write("This app uses the SpMV-CNN model to estimate runtime and energy consumption of sparse matrix-vector products.")

# File uploader
uploaded = st.file_uploader("Upload a sparse matrix file (.npy)", type=["npy"])

if uploaded is not None:
    st.success(f"File uploaded: {uploaded.name}")

    # Load the uploaded matrix
    matrix = np.load(uploaded)
    st.write("Matrix shape:", matrix.shape)

    # ---- Load model (adjust path as needed) ----
    # Example: assumes you saved a trained PyTorch model checkpoint in the repo as cnn_model.pth
    try:
        model = torch.load("cnn_model.pth", map_location=torch.device("cpu"))
        model.eval()

        # Convert to tensor
        x = torch.tensor(matrix, dtype=torch.float32)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # add batch dimension

        # Run prediction
        with torch.no_grad():
            prediction = model(x)

        st.subheader("Predicted Values")
        st.write("Runtime (ms):", float(prediction[0][0]))
        st.write("Energy Consumption (J):", float(prediction[0][1]))

    except Exception as e:
        st.error("Model file not found or inference failed.")
        st.code(str(e))
        st.info("Make sure a trained model checkpoint (cnn_model.pth) exists in the repo root.")
