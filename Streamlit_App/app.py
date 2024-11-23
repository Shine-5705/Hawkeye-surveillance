import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from datetime import datetime
import os
from twilio.rest import Client
import logging
import tempfile

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  handlers=[
      logging.FileHandler('fight_detection.log'),
      logging.StreamHandler()
  ]
)

# Twilio configuration
TWILIO_ACCOUNT_SID = st.secrets["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = st.secrets["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER = st.secrets["TWILIO_PHONE_NUMBER"]
EMERGENCY_NUMBER = st.secrets["EMERGENCY_NUMBER"]

# Metrics functions
def recall_m(y_true, y_pred):
  true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
  possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
  return recall

def precision_m(y_true, y_pred):
  true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
  predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
  return precision

def f1_m(y_true, y_pred):
  precision = precision_m(y_true, y_pred)
  recall = recall_m(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))

def send_emergency_alert(detection_type="webcam", location="Unknown"):
  try:
      client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
      
      message_text = (f"⚠️ EMERGENCY ALERT! Fight detected via {detection_type} "
                     f"at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                     f"Location: {location}")
      
      message = client.messages.create(
          body=message_text,
          from_=TWILIO_PHONE_NUMBER,
          to=EMERGENCY_NUMBER
      )
      
      call = client.calls.create(
          twiml=f'<Response><Say>Emergency! Fight detected via {detection_type}.</Say></Response>',
          from_=TWILIO_PHONE_NUMBER,
          to=EMERGENCY_NUMBER
      )
      
      return True
  except Exception as e:
      st.error(f"Error sending alert: {str(e)}")
      return False

def process_video_feed(frame, model, frames_buffer, segment_frames=42):
  try:
      resized_frame = cv2.resize(frame, (128, 128))
      frames_buffer.append(resized_frame)
      
      if len(frames_buffer) > segment_frames:
          frames_buffer.pop(0)
      
      if len(frames_buffer) == segment_frames:
          video_segment = np.array(frames_buffer)
          video_segment = video_segment.astype('float32') / 255.0
          video_segment = np.expand_dims(video_segment, axis=0)
          
          prediction = model.predict(video_segment, verbose=0)
          fight_prob = prediction[0][1]
          predicted_class = 1 if fight_prob > 0.4 else 0
          
          return predicted_class, fight_prob
          
      return None, 0.0
      
  except Exception as e:
      st.error(f"Error in processing: {str(e)}")
      return None, 0.0

def main():
  st.set_page_config(page_title="Fight Detection System", page_icon="🎥")
  st.title("Fight Detection System")

  # Sidebar
  st.sidebar.title("Settings")
  detection_mode = st.sidebar.radio("Select Detection Mode", 
                                  ["Webcam Detection", "Video File Detection", "Test Alert System"])
  
  confidence_threshold = st.sidebar.slider("Confidence Threshold", 
                                         min_value=0.1, 
                                         max_value=0.9, 
                                         value=0.4)

  # Load model
  @st.cache_resource
  def load_model():
      return keras.models.load_model(r"C:\Users\gupta\OneDrive\Desktop\INNOTECH\Real-Time-Violence-Detection\vivit_model", 
                                   custom_objects={'recall_m': recall_m, 
                                                 'precision_m': precision_m, 
                                                 'f1_m': f1_m})

  try:
      model = load_model()
  except Exception as e:
      st.error(f"Error loading model: {str(e)}")
      return

  if detection_mode == "Webcam Detection":
      st.header("Webcam Fight Detection")
      run = st.button("Start Detection")
      
      if run:
          frames_buffer = []
          last_alert_time = time.time()
          stframe = st.empty()
          status_text = st.empty()
          
          cap = cv2.VideoCapture(0)
          
          try:
              while run:
                  ret, frame = cap.read()
                  if not ret:
                      st.error("Error accessing webcam")
                      break
                  
                  pred_class, confidence = process_video_feed(frame, model, frames_buffer)
                  
                  if pred_class is not None:
                      current_time = time.time()
                      
                      if pred_class == 1 and confidence >= confidence_threshold:
                          frame = cv2.rectangle(frame, (0, 0), 
                                             (frame.shape[1], frame.shape[0]), 
                                             (0, 0, 255), 10)
                          
                          if current_time - last_alert_time >= 60:
                              send_emergency_alert("webcam")
                              last_alert_time = current_time
                              
                          status_text.warning("⚠️ FIGHT DETECTED!")
                      else:
                          status_text.info("No Fight Detected")
                  
                  # Display confidence
                  cv2.putText(frame, f"Confidence: {confidence:.2%}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                  
                  stframe.image(frame, channels="BGR")
                  
          finally:
              cap.release()

  elif detection_mode == "Video File Detection":
      st.header("Video File Fight Detection")
      uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
      
      if uploaded_file is not None:
          tfile = tempfile.NamedTemporaryFile(delete=False) 
          tfile.write(uploaded_file.read())
          
          frames_buffer = []
          last_alert_time = 0
          stframe = st.empty()
          status_text = st.empty()
          
          cap = cv2.VideoCapture(tfile.name)
          
          try:
              while cap.isOpened():
                  ret, frame = cap.read()
                  if not ret:
                      break
                  
                  pred_class, confidence = process_video_feed(frame, model, frames_buffer)
                  
                  if pred_class is not None:
                      current_time = time.time()
                      
                      if pred_class == 1 and confidence >= confidence_threshold:
                          frame = cv2.rectangle(frame, (0, 0), 
                                             (frame.shape[1], frame.shape[0]), 
                                             (0, 0, 255), 10)
                          
                          if current_time - last_alert_time >= 60:
                              send_emergency_alert("video")
                              last_alert_time = current_time
                              
                          status_text.warning("⚠️ FIGHT DETECTED!")
                      else:
                          status_text.info("No Fight Detected")
                  
                  cv2.putText(frame, f"Confidence: {confidence:.2%}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                  
                  stframe.image(frame, channels="BGR")
                  
          finally:
              cap.release()
              os.unlink(tfile.name)

  elif detection_mode == "Test Alert System":
      st.header("Test Alert System")
      if st.button("Send Test Alert"):
          if send_emergency_alert("test"):
              st.success("Test alert sent successfully!")
          else:
              st.error("Failed to send test alert")

if __name__ == "__main__":
  main()