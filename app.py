# Importing necessary Libraries
import asyncio 
from collections import deque
import numpy as np
import streamlit as st
import librosa 
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer , WebRtcMode
import av 
import mediapipe as mp
import cv2
import pandas as pd
####################################################### UI #####################################################################################

st.set_page_config(page_title="SpeakSmart + CalmCoreAI ( Real-Time Coach for Public Speaking and Interviews )",page_icon="🗣️",layout= "wide" )
st.title("SpeakSmart + CalmCoreAI ( Real-Time Coach for Public Speaking and Interviews )")
st.caption("Pitch Tracking + Confidence Analyzer + Webcam BodyLanguage Overlay")

#################################################### WebRtc Config ####################################################################################
RTC_Configuration = {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}

################################################### Session State  #####################################################################################
if "pitch_hist" not in st.session_state:
    st.session_state.pitch_hist = deque(maxlen=1200) 
if "rms_hist" not in st.session_state:
    st.session_state.rms_hist = deque(maxlen=1200)
if "timestamps" not in st.session_state:
    st.session_state.timestamps = deque(maxlen=1200)
if "last_ts" not in st.session_state:
    st.session_state.last_ts = 0.0
if "live_confidence" not in st.session_state:
    st.session_state.live_confidence = 0

################################################ Control ###############################################################################################

col1, col2, col3 = st.columns(3)
with col1:
    fmin = st.number_input("Min Pitch (Hz)", 60.0, 400.0, 80.0, 10.0)
with col2:
    fmax = st.number_input("Max Pitch (Hz)", 120.0, 800.0, 300.0, 10.0)
with col3:
    window_sec = st.slider("Confidence window (s)", 3, 20, 8)

silence_rms_thresh = st.slider(
    "Silence threshold (RMS)", 0.0, 0.05, 0.005, 0.001,
    help="Frames below this are ignored as silence."
)

################################################### Pitch Estimation ####################################################################################

HOP_DUR = 0.05  # ~50 ms

def estimate_pitch(y: np.ndarray, sr: int, fmin: float, fmax: float):
    """Compute median f0 for the frame using YIN. Return (f0, rms)."""
    if y.size < int(0.03 * sr):  
        return np.nan, 0.0

    rms = float(np.sqrt(np.mean(y**2)))
    if rms < silence_rms_thresh:
        return np.nan, rms

    try:
        f0 = librosa.yin(y.astype(np.float32), fmin=fmin, fmax=fmax, sr=sr)
        f0 = f0[np.isfinite(f0)]
        return (float(np.median(f0)), rms) if f0.size else (np.nan, rms)
    except Exception:
        return np.nan, rms
    
############################################### Chunker ###################################################################################################
class AudioChunker:
    def __init__(self, sr=48000, hop_dur=HOP_DUR):
        self.sr = sr
        self.hop = int(sr * hop_dur)
        self.buf = np.zeros(0, dtype=np.float32)

    def push(self, samples: np.ndarray):
        self.buf = np.concatenate([self.buf, samples])
        out = []
        while self.buf.size >= self.hop:
            out.append(self.buf[:self.hop])
            self.buf = self.buf[self.hop:]
        return out

chunker = AudioChunker()

########################################## Audio CallBack #####################################################################################################

def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    pcm = frame.to_ndarray()
    if pcm.ndim == 2:
        mono = pcm.mean(axis=0)
    else:
        mono = pcm 
    # normalize int16 → float32
    if np.issubdtype(mono.dtype, np.integer):
        mono = mono.astype(np.float32) / np.iinfo(mono.dtype).max
    else:
        mono = mono.astype(np.float32)

    for chunk in chunker.push(mono):
        f0, rms = estimate_pitch(chunk, sr=48000, fmin=fmin, fmax=fmax)
        ts = st.session_state.last_ts + HOP_DUR
        st.session_state.last_ts = ts
        st.session_state.timestamps.append(ts)
        st.session_state.rms_hist.append(rms)
        st.session_state.pitch_hist.append(np.nan if np.isnan(f0) else f0)
        st.session_state.live_confidence = calc_confidence_score()

    return frame

####################################### Confidence Analyzer ###################################################################################################

def calc_confidence_score():
    if not st.session_state.timestamps:
        return 0

    t_now = st.session_state.timestamps[-1]
    t_min = t_now - window_sec
    idx = [i for i, t in enumerate(st.session_state.timestamps) if t >= t_min]
    if not idx:
        return 0

    recent_f0 = np.array(list(st.session_state.pitch_hist))[idx[0]:]
    recent_rms = np.array(list(st.session_state.rms_hist))[idx[0]:]
    mask = (~np.isnan(recent_f0)) & (recent_rms >= silence_rms_thresh)
    voiced = recent_f0[mask]
    if voiced.size < 5:
        return 0

    f0_std = float(np.std(voiced))
    f0_med = float(np.median(voiced))
    rel_var = f0_std / max(f0_med, 1e-6)
    score = 100.0 * np.clip(1.0 - (rel_var / 0.25), 0.0, 1.0)
    return int(round(score))

############################################## Video #########################################################################################################

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

class VideoProcessor:
    def __init__(self):
        self.detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)

        if res.detections:
            for det in res.detections:
                mp_draw.draw_detection(img, det)

        # overlay tips
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (350, 110), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        cv2.putText(img, "Tips:", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(img, "Look at camera • Relax shoulders", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        cv2.putText(img, "Steady head • Natural smile", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
################################################# Layout ######################################################################################################

col1, col2 = st.columns(2)
with col1:
    st.subheader("Webcam")
    webrtc_streamer(
        key="video",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration =  RTC_Configuration,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Microphone (Pitch)")
    webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration= RTC_Configuration,
        audio_frame_callback=audio_frame_callback,
        media_stream_constraints={"video": False, "audio": True},
    )

################################################# Live Charts #################################################################################################

st.markdown("---")
lcol, rcol = st.columns(2)

with lcol:
    st.subheader("Live Pitch (Hz)")
    plot = st.line_chart()

with rcol:
    st.subheader("Live Confidence")
    conf_placeholder = st.empty()
    conf_placeholder.metric("confidence (0-100)",
    int(st.session_state.live_confidence))

async def ui_updater():
    while True:
        if st.session_state.timestamps:
            xs = list(st.session_state.timestamps)
            ys = list(st.session_state.pitch_hist)
            new_data = pd.DataFrame({"time": xs,"Pitch":[None if np.isnan(v) else  v for v in ys]})
            pd.add_rows(new_data)
            conf_metric.metric("Confidence (0–100)", value=st.session_state.live_confidence)
        await asyncio.sleep(0.2)

try:
    asyncio.get_running_loop().create_task(ui_updater())
except RuntimeError:
    pass

####################################################### Help Text #############################################################################################

st.markdown("""
**How to use:**
1. Allow camera & mic.
2. Speak for a few seconds — pitch will update every ~50ms.
3. Confidence updates as pitch stabilizes.

Notes:  
- Only pitch stability is used (not content).  
- Silence (RMS < threshold) is ignored.  
- Adjust pitch range if needed.  
""")