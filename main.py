import streamlit as st
import numpy as np
from hebb import Hebb

# *********************** Util Function
def add_data(training_data_input, target_input):
    st.session_state.target_data.append(target_input)
    st.session_state.training_data.append(training_data_input.flatten())

def create_model():
    training_data = np.array(st.session_state.training_data)
    target_data = np.array(st.session_state.target_data)
    HebbModel = Hebb(training_data, target_data)
    return HebbModel

def reset_checkboxes():
    st.session_state.checkbox_states = [[False for _ in range(5)] for _ in range(5)]
    for i in range(5):
        for j in range(5):
            key = f"tr_{i}_{j}"
            if key in st.session_state:
                st.session_state[key] = False

def reset_checkboxes_predict():
    st.session_state.checkbox_states_predict = [[False for _ in range(5)] for _ in range(5)]
    for i in range(5):
        for j in range(5):
            key = f"cb_{i}_{j}"
            if key in st.session_state:
                st.session_state[key] = False

def train_step(): 
    message = st.session_state.hebb_model.train_step()
    if message:
        st.session_state.model_converge = message
        

def train():
    message = st.session_state.hebb_model.train()
    if message:
        st.session_state.model_converge = message
    
def set_up_training_data():
    if not st.session_state.is_setup_training_data:
        st.session_state.is_setup_training_data = True
        st.session_state.hebb_model = create_model()

def predict():
    prediction = st.session_state.hebb_model.predict(grid.flatten())
    st.session_state.predict_result = prediction


# *********************** Streamlit Section
st.title("Simulasi Model Hebb untuk Pembelajaran Asosiatif")

#*************************** Training Data Section
st.header("Training Model Hebb")
st.subheader("Input Training Data")
if 'training_data' not in st.session_state:
    st.session_state.training_data = []
if 'target_data' not in st.session_state:
    st.session_state.target_data = []
if 'hebb_model' not in st.session_state:
    st.session_state.hebb_model = None  
if "is_setup_training_data" not in st.session_state:
    st.session_state.is_setup_training_data = False
if "model_converge" not in st.session_state:
    st.session_state.model_converge = None
if "predict_result" not in st.session_state:
    st.session_state.predict_result = None
if 'checkbox_states' not in st.session_state:
        st.session_state.checkbox_states = [[False for _ in range(5)] for _ in range(5)]
if 'checkbox_states_predict' not in st.session_state:
        st.session_state.checkbox_states_predict = [[False for _ in range(5)] for _ in range(5)]
training_data_input = np.full((5, 5), -1, dtype=int)

col = st.columns(2)
with col[0]:
    
    for i in range(5):
        cols = st.columns(5, gap="small") 
        for j in range(5):
            key = f"tr_{i}_{j}"
            st.session_state.checkbox_states[i][j] = cols[j].checkbox(
                "", 
                value=st.session_state.checkbox_states[i][j], 
                key=key
            )
            if st.session_state.checkbox_states[i][j]:
                training_data_input[i, j] = 1
            else:
                training_data_input[i, j] = -1

with col[1]:
    target_input = st.selectbox("Select Target", [-1, 1], placeholder="Select Target", key="target")
    
cols = st.columns([1, 1,3], gap="small")
with cols[0]:
    st.button("Tambah Data", on_click=add_data, args=(training_data_input, target_input), disabled=st.session_state.is_setup_training_data)
with cols[1]:
    st.button("Reset", on_click=reset_checkboxes, disabled=st.session_state.is_setup_training_data)

if len(st.session_state.training_data) >=2:
    st.button("Set Up Training Data", on_click=set_up_training_data, disabled=st.session_state.is_setup_training_data)

st.subheader("Training Data")
if st.session_state.training_data:
    reshaped_data = [np.array(data).reshape(5, 5) for data in st.session_state.training_data]
    for idx, data in enumerate(reshaped_data):
        cols = st.columns([2, 1, 1], gap="small")
        with cols[0]:
            st.write(f"Training Data {idx + 1}:")
            st.write(data)
        with cols[1]:
            st.write(f"Target Data {idx + 1}:")
            st.write(st.session_state.target_data[idx])
        if st.button(f"Delete", key=f"delete_{idx}", disabled=st.session_state.is_setup_training_data):
            del st.session_state.training_data[idx]
            del st.session_state.target_data[idx]
            st.rerun()
else:
    st.write("No Data")

#*************************** Model Information
st.subheader("Model Hebb Information")


if st.session_state.hebb_model:
    cols = st.columns(2)
    st.write("Model Name:", st.session_state.hebb_model.model_name)
    with cols[0]:
        st.write("Current Weights:", st.session_state.hebb_model.get_weights().reshape(5,5))
    with cols[1]:
        st.write("Current Data:", st.session_state.hebb_model.training_data[st.session_state.hebb_model.current_data].reshape(5,5))
    st.write("Bias:", st.session_state.hebb_model.get_bias())
    st.write("Epochs:", st.session_state.hebb_model.get_epoch())
else:
    st.write("Please setup training data first")

#*************************** Training Section
st.subheader("Train Model")
is_button_disable =  st.session_state.hebb_model is None or len(st.session_state.training_data) < 2

if is_button_disable:
    st.write("Minimal 2 data training untuk melatih model")

cols = st.columns([1,1,5], gap="small")
with cols[0]:
    st.button("Step", on_click=train_step, disabled=is_button_disable or bool(st.session_state.model_converge))
with cols[1]:
    st.button("Train", on_click=train, disabled=is_button_disable or bool(st.session_state.model_converge))

if st.session_state.model_converge:
    st.write(st.session_state.model_converge)

#*************************** Prediction Section
st.header("Model Test")
grid = np.full((5, 5), -1, dtype=int)
colsP = st.columns(2, gap="small")
with colsP[0]:
    for i in range(5):
        cols = st.columns(5, gap="small") 
        for j in range(5):
            key = f"cb_{i}_{j}"
            st.session_state.checkbox_states_predict[i][j] = cols[j].checkbox(
                "", 
                value=st.session_state.checkbox_states_predict[i][j], 
                key=key
            )
            if st.session_state.checkbox_states_predict[i][j]:
                grid[i, j] = 1 
            else:
                grid[i, j] = -1  

with colsP[1]:
    st.write("Test Data:")
    st.write(grid.flatten().reshape(5,5)) 
st.write("Prediction: ", st.session_state.predict_result)
cols = st.columns([1,1,4], gap="small")
with cols[0]:
    st.button("Predict", on_click=predict, disabled=is_button_disable)
with cols[1]:
    st.button("Reset", key="Reset Predict", on_click=reset_checkboxes_predict, disabled=is_button_disable)
