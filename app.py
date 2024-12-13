import streamlit as st
import numpy as np
import mne
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tempfile
import os
import pandas as pd
import PIL.Image

# Function to load the pre-trained model
def load_model(model_path="./unet_model_final.h5"):
    """Load the pre-trained model."""
    return tf.keras.models.load_model(model_path)

# Load the model once at the start
model = load_model()
def display_sleep_score_image(image_path):
    """
    Display sleep score or related image with controlled size
    
    Args:
        image_path (str): Path to the image file
    """
    try:
        # Open the image using PIL
        image = PIL.Image.open(image_path)
        
        # Resize image to a small, consistent size
        image.thumbnail((300, 300))  # Resize while maintaining aspect ratio
        
        # Create two columns to center the image
        col1, col2, col3 = st.columns([1,2,1])
        
        with col2:
            # Display the resized image
            st.image(image, 
                     caption='Sleep Analysis Preview', 
                     use_column_width='auto',
                     width=250)  # Explicitly set a small width
    except FileNotFoundError:
        st.warning(f"Image not found at {image_path}")
    except Exception as e:
        st.error(f"Error displaying image: {e}")

def load_and_preprocess_data1(uploaded_file):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        # Read raw EDF file and specify channel picks
        raw = mne.io.read_raw_edf(tmp_file_path, preload=True, verbose=False)

        # If no data channels are found, raise an error
        if len(raw.ch_names) == 0:
            raise ValueError("No data channels found in the EDF file.")

        # Optionally log the channel names
        print(f"Channel names: {raw.ch_names}")

        # Apply bandpass filter
        raw.filter(0.5, 30, fir_design='firwin')

        # Calculate epoch parameters
        sfreq = raw.info['sfreq']  # Sampling frequency
        epoch_duration = 30.0  # 30-second epochs
        epoch_samples = int(epoch_duration * sfreq)

        # Create epochs manually
        n_epochs = int(raw.n_times / epoch_samples)
        features = np.zeros((n_epochs, raw.info['nchan'], epoch_samples))

        # Segment raw data into 30-second epochs
        for i in range(n_epochs):
            start_sample = i * epoch_samples
            end_sample = start_sample + epoch_samples

            # Extract epoch data
            epoch_data = raw.get_data(
                start=start_sample,
                stop=min(end_sample, raw.n_times)
            )

            # Pad if the last epoch is shorter
            if epoch_data.shape[1] < epoch_samples:
                pad_width = ((0, 0), (0, epoch_samples - epoch_data.shape[1]))
                epoch_data = np.pad(epoch_data, pad_width, mode='constant')

            features[i] = epoch_data

        return features

    except Exception as e:
        raise ValueError(f"Failed to process the EDF file: {e}")

def predict_sleep_stages(data, model):
    """Predict sleep stages for preprocessed EEG data."""
    predictions = model.predict(data)
    sleep_stages = np.argmax(predictions, axis=1)  # Get the class with the highest probability
    return sleep_stages

def plot_sleep_stages(predicted_stages):
    # Map each sleep stage to a unique numeric value for plotting
    stage_mapping = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}
    
    # Map numeric predictions to sleep stage names
    stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    predicted_stage_names = [stage_map[stage] for stage in predicted_stages]
    
    numeric_stages = np.array([stage_mapping[stage] for stage in predicted_stage_names])

    # Create time points in hours (30-second epochs converted to hours)
    time_points = np.arange(len(predicted_stage_names)) * 30 / 3600  # Convert seconds to hours

    # Create the plot
    plt.figure(figsize=(20, 8))

    # Plot the continuous line for all stages
    plt.plot(time_points, numeric_stages, color='blue', linewidth=2)

    # Adding title and labels
    plt.title('Predicted Sleep Stages Over Time', fontsize=20)

    # X-ticks formatting with bold and larger text
    plt.xticks(fontsize=12, fontweight='bold')

    # X-axis label
    plt.xlabel('Time (hours)', fontsize=16)

    # Y-axis label and ticks
    plt.ylabel('Sleep Stages', fontsize=16)
    plt.yticks(list(stage_mapping.values()),
               [f'$\\mathbf{{{stage}}}$' for stage in stage_mapping.keys()],
               fontsize=12)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Optimize layout
    plt.tight_layout()
    
    return plt

def main():
    st.set_page_config(page_title="Sleep Stage Prediction", page_icon=":sleeping:")

    # Path to your sleep score image
    sleep_score_image_path = "./sleepnet.png"  # Update this path to your actual image

    # Try to display the sleep score image before the main content
    if os.path.exists(sleep_score_image_path):
        display_sleep_score_image(sleep_score_image_path)
    st.title("Sleep Stage Prediction")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an EDF file", type="edf")

    # Upload and process button
    if st.button("Upload and Process"):
        if uploaded_file is not None:
            # Load and preprocess data
            try:
                eeg_data = load_and_preprocess_data1(uploaded_file)
                
                # Scale the data
                scaler = StandardScaler()
                X_test_scaled = scaler.fit_transform(eeg_data.reshape(eeg_data.shape[0], -1)).reshape(eeg_data.shape)
                X_test_scaled = np.transpose(X_test_scaled, (0, 2, 1))
                
                # Predict sleep stages
                predictions = predict_sleep_stages(X_test_scaled, model)
                
                # Map numeric predictions to sleep stage names
                stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
                predicted_stages = [stage_map[stage] for stage in predictions]
                
                # Display stage distribution
                st.subheader("Sleep Stage Distribution")
                unique, counts = np.unique(predictions, return_counts=True)
                stage_dist = dict(zip([stage_map[u] for u in unique], counts))
                st.write(stage_dist)
                
                # Plot and display
                st.subheader("Predicted Sleep Stages ")
                plt_fig = plot_sleep_stages(predictions)
                st.pyplot(plt_fig)

                # Generate CSV for download
                st.subheader("Download Predicted Sleep Stages")
                epoch_data = {
                    "Epoch": np.arange(1, len(predictions) + 1),
                    "Predicted Stage": predicted_stages
                }
                df = pd.DataFrame(epoch_data)

                # Convert DataFrame to CSV
                csv = df.to_csv(index=False)

                # Provide download link
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="predicted_sleep_stages.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please upload a file before processing.")

if __name__ == "__main__":
    main()
