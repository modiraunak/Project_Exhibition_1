# Importing necessary Libraries
import asyncio 
from collections import deque
import numpy as np
import streamlit as st
import librosa 
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer , WebRtcMode , RTCConfiguration
import av 
import mediapipe as mp
import cv2

####################################################### UI #####################################################################################

st.set_page_config(page_title="SpeakSmart + CalmCoreAI ( Real-Time Coach for Public Speaking and Interviews )",page_icon="🗣️",layout= "wide" )
st.title("SpeakSmart + CalmCoreAI ( Real-Time Coach for Public Speaking and Interviews )")
st.caption("Pitch Tracking + Confidence Analyzer + Webcam BodyLanguage Overlay")

