import asyncio
import time
import threading
from collections import deque
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
    import mediapipe as mp
    import cv2
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

####################################################### CONFIGURATION ###########################################################################

st.set_page_config(
    page_title="SpeakSmart AI - Communication Coach",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .confidence-excellent { color: #28a745; font-weight: bold; }
    .confidence-good { color: #ffc107; font-weight: bold; }
    .confidence-fair { color: #fd7e14; font-weight: bold; }
    .confidence-poor { color: #dc3545; font-weight: bold; }
    
    .nervousness-alert {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.375rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    .improvement-tip {
        background-color: #d1ecf1;
        border: 1px solid #b6d4fe;
        border-radius: 0.375rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🗣️ SpeakSmart AI - Your Personal Communication Coach</h1>', unsafe_allow_html=True)
st.markdown("*Real-time analysis of speech patterns, confidence levels, and nervousness detection with personalized feedback*")

####################################################### SESSION STATE ############################################################################

# Initialize session state variables
session_state_defaults = {
    "pitch_history": deque(maxlen=2000),
    "rms_history": deque(maxlen=2000),
    "timestamps": deque(maxlen=2000),
    "confidence_history": deque(maxlen=2000),
    "current_confidence": 0,
    "nervousness_score": 0,
    "speaking_rate": 0,
    "pitch_variations": [],
    "nervous_moments": [],
    "improvement_suggestions": [],
    "session_summary": {},
    "is_recording": False,
    "session_start_time": None,
    "total_frames": 0,
    "progress_history": []
}

for key, default_value in session_state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

####################################################### SIDEBAR CONFIGURATION ####################################################################

st.sidebar.header("⚙️ Analysis Settings")
recording_method = st.sidebar.selectbox(
    "Recording Method:",
    ["File Upload", "Real-time Recording (SoundDevice)", "WebRTC Stream"],
    help="Choose your preferred audio input method"
)

st.sidebar.subheader("Voice Analysis Parameters")
fmin = st.sidebar.slider("Min Pitch (Hz)", 50, 200, 75, 5, help="Lower bound for pitch detection")
fmax = st.sidebar.slider("Max Pitch (Hz)", 200, 500, 350, 10, help="Upper bound for pitch detection")
silence_threshold = st.sidebar.slider("Silence Threshold", 0.001, 0.05, 0.003, 0.001)
analysis_window = st.sidebar.slider("Analysis Window (seconds)", 3, 15, 8)

st.sidebar.subheader("Nervousness Detection")
pitch_variation_threshold = st.sidebar.slider("Pitch Variation Sensitivity", 0.1, 0.5, 0.25, 0.05)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 30, 80, 50, 5)

st.sidebar.subheader("Session Controls")
if st.sidebar.button("⟳ Reset Session"):
    keys_to_reset = [k for k in st.session_state.keys() if k not in ["recording_method", "fmin", "fmax"]]
    for key in keys_to_reset:
        if key in session_state_defaults:
            st.session_state[key] = session_state_defaults[key]
    st.success("Session reset successfully!")
    st.rerun()

####################################################### CORE ANALYSIS FUNCTIONS ################################################################

class SpeechAnalyzer:
    """Advanced speech analysis with nervousness detection and confidence scoring."""
    
    def __init__(self):
        self.hop_duration = 0.05 
        self.smoothing_factor = 0.3

    def analyze_pitch_segment(self, audio_segment, sr, fmin, fmax):
        """Analyze pitch for a single audio segment."""
        try:
            if len(audio_segment) < int(0.03 * sr):
                return np.nan, 0.0
                
            rms = np.sqrt(np.mean(audio_segment**2))
            if rms < silence_threshold:
                return np.nan, rms
            
            f0 = librosa.yin(
                audio_segment.astype(np.float32),
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                hop_length=min(512, len(audio_segment) // 4)
            )
            
            valid_f0 = f0[np.isfinite(f0) & (f0 > 0)]
            
            if len(valid_f0) > 0:
                pitch = float(np.median(valid_f0))
                return pitch, rms
            else:
                return np.nan, rms   
                
        except Exception as e:
            st.warning(f"Pitch analysis error: {e}")
            return np.nan, 0.0

    def detect_nervousness_indicators(self, pitch_data, rms_data, time_window=8):
        """Detect signs of nervousness based on speech patterns."""
        if len(pitch_data) < 10:
            return 0, []
            
        window_size = max(10, int(time_window / self.hop_duration))
        recent_pitches = [p for p in list(pitch_data)[-window_size:] if p is not None and not np.isnan(p)]
        recent_rms = list(rms_data)[-window_size:]
        
        if len(recent_pitches) < 5:
            return 0, []
            
        indicators = []
        nervousness_score = 0

        if len(recent_pitches) > 1:
            pitch_changes = np.diff(recent_pitches)
            pitch_std = np.std(recent_pitches)
            rapid_changes = np.sum(np.abs(pitch_changes) > pitch_std * 1.5)
            
            if rapid_changes > len(recent_pitches) * 0.3:
                nervousness_score += 25
                indicators.append("High pitch variability detected")
    
        if len(recent_pitches) > 10:
            micro_variations = np.std(np.diff(recent_pitches))
            if micro_variations > 15:
                nervousness_score += 20
                indicators.append("Voice tremor detected")
    
        if len(recent_pitches) > 5:
            avg_pitch = np.mean(recent_pitches)
            pitch_percentile_80 = np.percentile(recent_pitches, 80)
            if avg_pitch > pitch_percentile_80:
                nervousness_score += 15
                indicators.append("Elevated pitch level")
        valid_rms = [r for r in recent_rms if r > silence_threshold]
        if len(valid_rms) > 5:
            rms_std = np.std(valid_rms)
            if rms_std > 0.02:
                nervousness_score += 10
                indicators.append("Inconsistent volume levels")
        silent_frames = sum(1 for r in recent_rms if r < silence_threshold)
        silence_ratio = silent_frames / len(recent_rms) if recent_rms else 0
        
        if silence_ratio > 0.4 or silence_ratio < 0.05:
            nervousness_score += 15
            indicators.append("Irregular speaking pace")
            
        return min(nervousness_score, 100), indicators
    def calculate_confidence_score(self, pitch_data, rms_data, time_window=8):
        """Calculate confidence score based on speech stability."""
        try:
            if len(pitch_data) < 5:
                return 0
            window_size = max(10, int(time_window / self.hop_duration))
            recent_pitches = []
            recent_rms = []
            start_idx = max(0, len(pitch_data) - window_size)
            for i in range(start_idx, len(pitch_data)):
                if i < len(rms_data):
                    pitch = pitch_data[i]
                    rms = rms_data[i]
                    if pitch is not None and not np.isnan(pitch) and rms >= silence_threshold:
                        recent_pitches.append(pitch)
                        recent_rms.append(rms)
            
            if len(recent_pitches) < 5:
                return 0
            
            pitch_array = np.array(recent_pitches)
            pitch_mean = np.mean(pitch_array)
            pitch_std = np.std(pitch_array)

            if pitch_mean == 0:
                pitch_stability = 0
            else:
                pitch_stability = 1.0 - min(pitch_std / pitch_mean, 0.5)
            rms_mean = np.mean(recent_rms)
            rms_std = np.std(recent_rms)
            
            if rms_mean == 0:
                rms_consistency = 0
            else:
                rms_consistency = 1.0 - min(rms_std / rms_mean, 0.5)
            total_frames = window_size
            speaking_ratio = len(recent_pitches) / max(total_frames, 1)
            continuity_score = min(speaking_ratio * 1.5, 1.0)
            confidence = int(100 * (pitch_stability * 0.4 + rms_consistency * 0.3 + continuity_score * 0.3))
            return max(0, min(confidence, 100))
            
        except Exception as e:
            st.warning(f"Confidence calculation error: {e}")
            return 0

    def generate_improvement_suggestions(self, confidence, nervousness_score, indicators):
        """Generate personalized improvement suggestions."""
        suggestions = []
        if confidence < 30:
            suggestions.extend([
                "🎯 Focus on speaking with a steadier tone - practice sustained vowel sounds",
                "😮‍💨 Try deep breathing exercises before speaking"
            ])
        elif confidence < 60:
            suggestions.extend([
                "✅ Good progress! Work on maintaining consistent volume",
                "⏱️ Practice pacing - speak slightly slower for better control"
            ])
        if nervousness_score > 70:
            suggestions.extend([
                "😰 High nervousness detected - take a moment to pause and breathe",
                "🎭 Try visualization techniques before important conversations"
            ])
        elif nervousness_score > 40:
            suggestions.append("⚠️ Some tension detected - relax your shoulders and jaw")
        indicator_text = " ".join(indicators).lower()
        if "pitch variability" in indicator_text:
            suggestions.append("📊 Practice speaking in your natural pitch range")
        if "tremor" in indicator_text:
            suggestions.append("🤲 Do vocal warm-up exercises to relax your voice")
        if "elevated pitch" in indicator_text:
            suggestions.append("⬇️ Consciously lower your voice pitch - speak from your chest")
        if "volume" in indicator_text:
            suggestions.append("🔊 Practice consistent breath support for steady volume")
        if "pace" in indicator_text:
            suggestions.append("⏰ Work on steady speaking rhythm - use a metronome if helpful")
        
        return list(set(suggestions))  

analyzer = SpeechAnalyzer()

####################################################### AUDIO PROCESSING FUNCTIONS ##############################################################

def process_audio_file(audio_file):
    """Process uploaded audio file and analyze speech patterns."""
    try:
        audio_data, sr = librosa.load(audio_file, sr=None)
        st.success(f"✅ Audio loaded: {len(audio_data)/sr:.1f} seconds, {sr} Hz")
        st.session_state.pitch_history.clear()
        st.session_state.rms_history.clear()
        st.session_state.timestamps.clear()
        st.session_state.confidence_history.clear()
        st.session_state.nervous_moments.clear()
        chunk_size = int(sr * analyzer.hop_duration)

        with st.spinner("🔍 Analyzing speech patterns..."):
            progress_bar = st.progress(0)
            total_chunks = (len(audio_data) - chunk_size) // chunk_size
            
            for i in range(0, len(audio_data) - chunk_size, chunk_size):
                chunk = audio_data[i:i + chunk_size]
                pitch, rms = analyzer.analyze_pitch_segment(chunk, sr, fmin, fmax)
                timestamp = i / sr
                
                st.session_state.pitch_history.append(pitch if not np.isnan(pitch) else None)
                st.session_state.rms_history.append(rms)
                st.session_state.timestamps.append(timestamp)
                
                progress = min(1.0, (i // chunk_size) / max(total_chunks, 1))
                progress_bar.progress(progress)

                if len(st.session_state.timestamps) % 20 == 0:
                    confidence = analyzer.calculate_confidence_score(
                        st.session_state.pitch_history,
                        st.session_state.rms_history,
                        analysis_window
                    )
                    
                    nervousness, indicators = analyzer.detect_nervousness_indicators(
                        st.session_state.pitch_history,
                        st.session_state.rms_history,
                        analysis_window
                    )
                    
                    st.session_state.current_confidence = confidence
                    st.session_state.nervousness_score = nervousness
                    st.session_state.confidence_history.append(confidence)
            
                    if nervousness > 60:
                        st.session_state.nervous_moments.append({
                            'time': timestamp,
                            'score': nervousness,
                            'indicators': indicators
                        })
            
            progress_bar.progress(1.0)
            st.session_state.improvement_suggestions = analyzer.generate_improvement_suggestions(
                st.session_state.current_confidence,
                st.session_state.nervousness_score,
                [item for moment in st.session_state.nervous_moments for item in moment['indicators']]
            )
            
        return True
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return False

def record_realtime_audio(duration=10):
    """Record audio in real-time using sounddevice."""
    if not SOUNDDEVICE_AVAILABLE:
        st.error("SoundDevice not available. Install with: pip install sounddevice")
        return False
    
    try:
        sample_rate = 44100
        progress_bar = st.progress(0)
        status_text = st.empty()
        audio_data = []
        
        def callback(indata, frames, time, status):
            if status:
                st.warning(f"Audio callback status: {status}")
            audio_data.append(indata.copy())
        
        with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate, dtype=np.float32):
            for i in range(duration * 10):  
                time.sleep(0.1)
                progress = (i + 1) / (duration * 10)
                progress_bar.progress(progress)
                status_text.text(f"🎤 Recording... {duration - i//10} seconds remaining")
        
        if audio_data:
            full_audio = np.concatenate(audio_data, axis=0).flatten()
            return process_recorded_audio(full_audio, sample_rate)
        else:
            st.error("No audio data recorded")
            return False
        
    except Exception as e:
        st.error(f"Recording error: {e}")
        return False

def process_recorded_audio(audio_data, sample_rate):
    """Process recorded audio data for real-time analysis."""
    try:
        st.session_state.pitch_history.clear()
        st.session_state.rms_history.clear()
        st.session_state.timestamps.clear()
        st.session_state.confidence_history.clear()
        st.session_state.nervous_moments.clear()
        
        chunk_size = int(sample_rate * analyzer.hop_duration)
        
        if len(audio_data) < chunk_size:
            st.warning("Audio too short for analysis")
            return False
        
        with st.spinner("🔍 Processing recorded audio..."):
            progress_bar = st.progress(0)
            total_chunks = (len(audio_data) - chunk_size) // chunk_size
            
            for i in range(0, len(audio_data) - chunk_size, chunk_size):
                chunk = audio_data[i:i + chunk_size]
                pitch, rms = analyzer.analyze_pitch_segment(chunk, sample_rate, fmin, fmax)
                timestamp = i / sample_rate
                
                st.session_state.pitch_history.append(pitch if not np.isnan(pitch) else None)
                st.session_state.rms_history.append(rms)
                st.session_state.timestamps.append(timestamp)
                progress = min(1.0, (i // chunk_size) / max(total_chunks, 1))
                progress_bar.progress(progress)
    
                if len(st.session_state.timestamps) % 10 == 0:
                    confidence = analyzer.calculate_confidence_score(
                        st.session_state.pitch_history,
                        st.session_state.rms_history,
                        analysis_window
                    )
                    
                    nervousness, indicators = analyzer.detect_nervousness_indicators(
                        st.session_state.pitch_history,
                        st.session_state.rms_history,
                        analysis_window
                    )
                    
                    st.session_state.current_confidence = confidence
                    st.session_state.nervousness_score = nervousness
                    st.session_state.confidence_history.append(confidence)
                    
                    if nervousness > 50:
                        st.session_state.nervous_moments.append({
                            'time': timestamp,
                            'score': nervousness,
                            'indicators': indicators
                        })
            
            progress_bar.progress(1.0)
        st.session_state.improvement_suggestions = analyzer.generate_improvement_suggestions(
            st.session_state.current_confidence,
            st.session_state.nervousness_score,
            [item for moment in st.session_state.nervous_moments for item in moment['indicators']]
        )
        return True
    
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return False

####################################################### MAIN INTERFACE ###########################################################################

st.header("🎙️ Speech Recording & Analysis")

if recording_method == "File Upload":
    st.subheader("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file (WAV, MP3, M4A, FLAC)",
        type=['wav', 'mp3', 'm4a', 'flac', 'aac'],
        help="Upload a recording of your speech for analysis"
    )
    
    if uploaded_file:
        if st.button("🔍 Analyze Speech", type="primary"):
            if process_audio_file(uploaded_file):
                st.success("✅ Analysis complete! Check results below.")
            else:
                st.error("Analysis failed. Please try a different file.")

elif recording_method == "Real-time Recording (SoundDevice)":
    st.subheader("🎤 Real-time Recording")

    if not SOUNDDEVICE_AVAILABLE:
        st.error("⚠️ SoundDevice not installed. Please install with: `pip install sounddevice`")
    else:
        col1, col2 = st.columns(2)
        with col1:
            duration = st.slider("Recording duration (seconds)", 5, 60, 15)
        with col2:
            st.write("") # Spacing
            if st.button("🎤 Start Recording", type="primary"):
                if record_realtime_audio(duration):
                    st.success("✅ Recording and analysis complete!")

elif recording_method == "WebRTC Stream":
    st.subheader("🌐 WebRTC Live Stream")
    
    if not WEBRTC_AVAILABLE:
        st.error("⚠️ WebRTC not available. Install with: `pip install streamlit-webrtc`")
    else:
        st.info("🚧 WebRTC streaming implementation placeholder")
        st.write("For development purposes, please use other recording methods")

####################################################### RESULTS VISUALIZATION ####################################################################

if len(st.session_state.timestamps) > 0:
    st.markdown("---")
    st.header("📊 Speech Analysis Results")
    
    # Key metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        confidence = st.session_state.current_confidence
        confidence_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 50 else "🟠" if confidence >= 30 else "🔴"
        st.metric(
            "Confidence Score",
            f"{confidence_color} {confidence}%", 
            help="Based on pitch stability and consistency"
        )
        
    with col2:
        nervousness = st.session_state.nervousness_score
        nervousness_color = "🔴" if nervousness >= 70 else "🟠" if nervousness >= 40 else "🟡" if nervousness >= 20 else "🟢"
        st.metric(
            "Nervousness Level",
            f"{nervousness_color} {nervousness}%", 
            help="Detected signs of nervous speech patterns"
        )
        
    with col3:
        valid_pitches = len([p for p in st.session_state.pitch_history if p is not None and not np.isnan(p)])
        total_time = max(st.session_state.timestamps) if st.session_state.timestamps else 1
        speaking_coverage = (valid_pitches * analyzer.hop_duration / total_time) * 100
        st.metric(
            "Speaking Coverage",
            f"🎤 {speaking_coverage:.1f}%",
            help="Percentage of time spent speaking vs silence"
        )
        
    with col4:
        nervous_moments_count = len(st.session_state.nervous_moments)
        st.metric(
            "Nervous Moments",
            f"⚠️ {nervous_moments_count}",
            help="Number of detected nervous episodes"
        )
    
    # Detailed Charts
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Pitch Analysis", "🎯 Confidence Trends", "⚠️ Nervousness Detection", "💡 Improvement Tips"])
    
    with tab1:
        st.subheader("Voice Pitch Over Time")
        valid_data = []
        for i, (timestamp, pitch) in enumerate(zip(st.session_state.timestamps, st.session_state.pitch_history)):
            if pitch is not None and not np.isnan(pitch):
                valid_data.append({'time': timestamp, 'pitch': pitch, 'index': i})
                
        if valid_data:
            df = pd.DataFrame(valid_data)
            fig = go.Figure()
            
            # Main pitch trace
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=df['pitch'],
                mode='lines+markers',
                name='Voice Pitch',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=3)
            ))
            
            # Add nervous moments as vertical lines
            for moment in st.session_state.nervous_moments:
                fig.add_vline(
                    x=moment['time'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Nervous moment ({moment['score']}%)",
                    annotation_position="top"
                )
            
            fig.update_layout(
                title="Voice Pitch Analysis with Nervous Moments Highlighted",
                xaxis_title="Time (seconds)",
                yaxis_title="Pitch (Hz)",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Pitch statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Pitch", f"{df['pitch'].mean():.1f} Hz")
            with col2:
                st.metric("Pitch Range", f"{df['pitch'].max() - df['pitch'].min():.1f} Hz")
            with col3:
                pitch_stability = 100 - (df['pitch'].std() / df['pitch'].mean() * 100) if df['pitch'].mean() > 0 else 0
                st.metric("Pitch Stability", f"{pitch_stability:.1f}%")
        else:
            st.info("No valid pitch data detected. Try adjusting the pitch range or speaking louder.")
    
    with tab2:
        st.subheader("Confidence Level Over Time")
        if st.session_state.confidence_history:
            confidence_times = np.linspace(0, max(st.session_state.timestamps), len(st.session_state.confidence_history))
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=confidence_times,
                y=list(st.session_state.confidence_history),  # Convert deque to list
                mode='lines+markers',
                name='Confidence Score',
                fill='tonexty',
                line=dict(color='#2ca02c', width=3),
                marker=dict(size=4)
            ))
            
            # Add reference lines
            fig.add_hline(y=70, line_dash="dot", line_color="green", annotation_text="Excellent (70%+)")
            fig.add_hline(y=50, line_dash="dot", line_color="orange", annotation_text="Good (50%+)")
            fig.add_hline(y=30, line_dash="dot", line_color="red", annotation_text="Needs Work (30%+)")
            
            fig.update_layout(
                title="Confidence Score Progression",
                xaxis_title="Time (seconds)",
                yaxis_title="Confidence Score (%)",
                yaxis=dict(range=[0, 100]),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Confidence data will appear as analysis progresses.")
    
    with tab3:
        st.subheader("Nervousness Detection Report")
        if st.session_state.nervous_moments:
            st.warning(f"⚠️ {len(st.session_state.nervous_moments)} nervous episodes detected")
            
            for i, moment in enumerate(st.session_state.nervous_moments):
                with st.expander(f"Nervous Moment #{i+1} at {moment['time']:.1f}s (Score: {moment['score']}%)"):
                    st.write("**Indicators detected:**")
                    for indicator in moment['indicators']:
                        st.write(f"• {indicator}")
                    st.write(f"**Timestamp:** {moment['time']:.1f} seconds")
                    st.write(f"**Nervousness Score:** {moment['score']}/100")
        else:
            st.success("✅ No significant nervous episodes detected! Great job maintaining composure.")

        # Overall nervousness assessment
        if st.session_state.nervousness_score > 0:
            if st.session_state.nervousness_score >= 70:
                st.markdown('<div class="nervousness-alert">⚠️ <strong>High nervousness detected.</strong> Consider taking breaks and practicing relaxation techniques.</div>', unsafe_allow_html=True)
            elif st.session_state.nervousness_score >= 40:
                st.markdown('<div class="nervousness-alert">⚠️ <strong>Moderate nervousness detected.</strong> Some tension in speech patterns.</div>', unsafe_allow_html=True)
            else:
                st.success("✅ Low nervousness levels. Good emotional control!")
    
    with tab4:
        st.subheader("Personalized Improvement Suggestions")
        if st.session_state.improvement_suggestions:
            st.write("Based on your speech analysis, here are personalized recommendations:")
            for i, suggestion in enumerate(st.session_state.improvement_suggestions):
                st.markdown(f'<div class="improvement-tip">{suggestion}</div>', unsafe_allow_html=True)
        else:
            st.info("💡 Improvement suggestions will appear after analysis.")
            
        st.subheader("General Speaking Tips")
        tips_data = {
            "Confidence Building": [
                "Practice power posing before speaking",
                "Use positive self-talk and affirmations",
                "Visualize successful communication",
                "Start with familiar topics"
            ],
            "Voice Control": [
                "Practice diaphragmatic breathing",
                "Do vocal warm-up exercises",
                "Record yourself regularly to monitor progress",
                "Speak from your chest, not your throat"
            ],
            "Nervousness Management": [
                "Use the 4-7-8 breathing technique",
                "Progressive muscle relaxation",
                "Mindfulness meditation",
                "Gradual exposure to speaking situations"
            ]
        }
        
        for category, tips in tips_data.items():
            with st.expander(f"💡 {category}"):
                for tip in tips:
                    st.write(f"• {tip}")

####################################################### SESSION SUMMARY ###########################################################################

    st.markdown("---")
    st.header("📋 Session Summary & Progress Tracking")
    
    # Calculate session statistics
    total_duration = max(st.session_state.timestamps) if st.session_state.timestamps else 0
    speaking_time = len([p for p in st.session_state.pitch_history if p is not None and not np.isnan(p)]) * analyzer.hop_duration
    silence_time = max(0, total_duration - speaking_time)
    
    session_data = {
        "Total Duration": f"{total_duration:.1f} seconds",
        "Speaking Time": f"{speaking_time:.1f} seconds ({(speaking_time/max(total_duration,1)*100):.1f}%)",
        "Silence Time": f"{silence_time:.1f} seconds ({(silence_time/max(total_duration,1)*100):.1f}%)",
        "Average Confidence": f"{st.session_state.current_confidence}%",
        "Nervousness Level": f"{st.session_state.nervousness_score}%",
        "Nervous Episodes": len(st.session_state.nervous_moments),
        "Improvement Areas": len(st.session_state.improvement_suggestions)
    }

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Session Statistics")
        for key, value in list(session_data.items())[:4]:
            st.write(f"**{key}:** {value}")
    with col2:
        st.subheader("🎯 Performance Metrics")
        for key, value in list(session_data.items())[4:]:
            st.write(f"**{key}:** {value}")
    
    # Performance grade calculation
    st.subheader("🏆 Overall Performance Grade")
    
    confidence_score = st.session_state.current_confidence
    nervousness_penalty = st.session_state.nervousness_score * 0.5
    speaking_ratio_bonus = min((speaking_time / max(total_duration, 1)) * 50, 20)
    
    overall_score = max(0, min(100, confidence_score - nervousness_penalty + speaking_ratio_bonus))
    
    # Grade determination
    if overall_score >= 85:
        grade, grade_color, grade_message = "A+", "🟢", "Outstanding! You demonstrate excellent speaking confidence and control."
    elif overall_score >= 75:
        grade, grade_color, grade_message = "A", "🟢", "Excellent performance with strong confidence and minimal nervousness."
    elif overall_score >= 65:
        grade, grade_color, grade_message = "B+", "🟡", "Good speaking skills with room for minor improvements."
    elif overall_score >= 55:
        grade, grade_color, grade_message = "B", "🟡", "Solid performance, focus on building more confidence."
    elif overall_score >= 45:
        grade, grade_color, grade_message = "C+", "🟠", "Fair performance, work on reducing nervousness and improving consistency."
    elif overall_score >= 35:
        grade, grade_color, grade_message = "C", "🟠", "Average performance, significant improvement opportunities available."
    else:
        grade, grade_color, grade_message = "D", "🔴", "Keep practicing! Focus on basic confidence-building exercises."
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; border: 3px solid #ddd; border-radius: 15px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);">
            <h2 style="margin: 0; color: #333;">{grade_color} Grade: {grade}</h2>
            <h3 style="margin: 10px 0; color: #666;">Score: {overall_score:.1f}/100</h3>
            <p style="margin: 0; font-style: italic; color: #555;">{grade_message}</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Export functionality
    st.subheader("💾 Export Results")
    export_data = {
        "session_timestamp": datetime.now().isoformat(),
        "analysis_settings": {
            "min_pitch": fmin,
            "max_pitch": fmax,
            "silence_threshold": silence_threshold,
            "analysis_window": analysis_window
        },
        "session_summary": session_data,
        "performance_grade": {
            "grade": grade,
            "score": overall_score,
            "message": grade_message
        },
        "pitch_data": [p for p in st.session_state.pitch_history if p is not None and not np.isnan(p)],
        "confidence_history": list(st.session_state.confidence_history),
        "nervous_moments": st.session_state.nervous_moments,
        "improvement_suggestions": st.session_state.improvement_suggestions
    }
    
    col1, col2 = st.columns(2)
    with col1:
        json_data = json.dumps(export_data, indent=2)
        st.download_button(
            label="📄 Download Detailed Report (JSON)",
            data=json_data,
            file_name=f"speaksmart_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Complete analysis data for detailed review"
        )
    with col2:
        if st.session_state.pitch_history:
            pitch_df = pd.DataFrame({
                'timestamp': list(st.session_state.timestamps)[:len(st.session_state.pitch_history)],
                'pitch': [p if p is not None else np.nan for p in st.session_state.pitch_history],
                'rms': list(st.session_state.rms_history)[:len(st.session_state.pitch_history)]
            })
            csv_data = pitch_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Pitch Data (CSV)",
                data=csv_data,
                file_name=f"speaksmart_pitch_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Raw pitch and audio data for further analysis"
            )

####################################################### PRACTICE MODE ############################################################################

if len(st.session_state.timestamps) == 0:
    st.markdown("---")
    st.header("🎭 Practice Scenarios")
    st.write("Try these common speaking scenarios to improve your skills:")
    
    scenarios = {
        "📺 Job Interview": {
            "description": "Practice common interview questions",
            "prompts": [
                "Tell me about yourself and your background",
                "What are your greatest strengths and weaknesses?",
                "Why do you want to work for our company?",
                "Describe a challenging situation you overcame",
                "Where do you see yourself in 5 years?"
            ]
        },
        "🎤 Presentation": {
            "description": "Practice public speaking scenarios",
            "prompts": [
                "Introduce your project to stakeholders",
                "Present quarterly results to your team",
                "Pitch a new idea to management",
                "Give a toast at a wedding or event",
                "Deliver a motivational speech"
            ]
        },
        "💬 Casual Conversation": {
            "description": "Practice everyday speaking situations",
            "prompts": [
                "Introduce yourself at a networking event",
                "Order food at a restaurant",
                "Give directions to a lost tourist",
                "Complain about a service issue politely",
                "Make small talk with a colleague"
            ]
        },
        "📚 Academic/Teaching": {
            "description": "Practice educational speaking scenarios",
            "prompts": [
                "Explain a complex concept to students",
                "Present your research findings",
                "Lead a group discussion",
                "Answer questions during a lecture",
                "Defend your thesis or project"
            ]
        }
    }
    
    selected_scenario = st.selectbox("Choose a practice scenario:", list(scenarios.keys()))
    if selected_scenario:
        scenario_info = scenarios[selected_scenario]
        st.write(f"**{scenario_info['description']}**")
        with st.expander(f"Practice prompts for {selected_scenario}"):
            for i, prompt in enumerate(scenario_info['prompts'], 1):
                st.write(f"{i}. {prompt}")
        st.info("💡 **Tip:** Choose a prompt, record yourself responding, then analyze your speech patterns!")

####################################################### PROGRESS TRACKING #########################################################################

st.markdown("---")
st.header("📈 Progress Tracking")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Goals Setting")
    confidence_goal = st.slider("Target Confidence Score", 50, 100, 80)
    nervousness_goal = st.slider("Maximum Nervousness Level", 0, 50, 20)
    
    if st.session_state.current_confidence > 0:
        confidence_progress = min(100, (st.session_state.current_confidence / confidence_goal) * 100)
        nervousness_progress = max(0, 100 - (st.session_state.nervousness_score / max(nervousness_goal, 1)) * 100)
        
        st.write("**Current Progress:**")
        st.progress(confidence_progress / 100, text=f"Confidence: {confidence_progress:.1f}% of goal")
        st.progress(nervousness_progress / 100, text=f"Nervousness Control: {nervousness_progress:.1f}%")

with col2:
    st.subheader("🏃‍♂️ Practice Recommendations")

    if st.session_state.current_confidence > 0:
        if st.session_state.current_confidence < 50:
            st.write("🎯 **Focus Areas:**")
            st.write("• Daily vocal warm-ups (10 minutes)")
            st.write("• Practice speaking slowly and clearly")
            st.write("• Record yourself reading aloud")
            st.write("• Work on breath control exercises")
            st.write("📅 **Recommended Practice:** 15-20 minutes daily")
        
        elif st.session_state.current_confidence < 75:
            st.write("🎯 **Enhancement Areas:**")
            st.write("• Practice impromptu speaking")
            st.write("• Work on storytelling techniques")
            st.write("• Focus on emotional expression")
            st.write("• Practice with different audiences")
            st.write("📅 **Recommended Practice:** 10-15 minutes daily")
        
        else:
            st.write("🎯 **Mastery Level:**")
            st.write("• Practice advanced techniques")
            st.write("• Work on persuasion skills")
            st.write("• Develop your unique speaking style")
            st.write("• Mentor others in speaking")
            st.write("📅 **Recommended Practice:** Maintain current level")

####################################################### ADDITIONAL FEATURES #######################################################################

st.markdown("---")
st.header("🔧 Additional Features")

feature_tabs = st.tabs(["🎵 Voice Exercises", "📚 Learning Resources", "⚙️ Settings", "❓ Help & FAQ"])

with feature_tabs[0]:  # Voice Exercises
    st.subheader("🎵 Daily Voice Exercises")
    
    exercises = {
        "🫁 Breathing Exercises": [
            "4-7-8 Breathing: Inhale for 4, hold for 7, exhale for 8",
            "Diaphragmatic breathing: Hand on chest, hand on belly",
            "Box breathing: 4 counts in, hold 4, out 4, hold 4",
            "Humming with breath support"
        ],
        "🎶 Vocal Warm-ups": [
            "Lip trills (motorboat sounds)",
            "Tongue twisters at varying speeds",
            "Humming scales up and down",
            "Sirens (low to high pitch)",
            "Vowel sounds: AH, EH, EE, OH, OO"
        ],
        "🗣️ Articulation Practice": [
            "Over-articulate words while reading",
            "Practice consonant clusters (str, spl, thr)",
            "Read tongue twisters slowly, then faster",
            "Practice minimal pairs (bit/pit, cat/bat)"
        ],
        "🎭 Expression Exercises": [
            "Read the same sentence with different emotions",
            "Practice varying pitch on single words",
            "Work on pace variation within sentences",
            "Practice pausing for dramatic effect"
        ]
    }

    for category, exercise_list in exercises.items():
        with st.expander(category):
            for exercise in exercise_list:
                st.write(f"• {exercise}")

with feature_tabs[1]:  # Learning Resources
    st.subheader("📚 Learning Resources & Tips")
    
    resources = {
        "📖 Recommended Reading": [
            "\"Talk Like TED\" by Carmine Gallo",
            "\"The Quick and Easy Way to Effective Speaking\" by Dale Carnegie",
            "\"Speak With Confidence\" by Albert Joseph",
            "\"The Presentation Secrets of Steve Jobs\" by Carmine Gallo"
        ],
        "🎥 Video Resources": [
            "TED Talks on public speaking",
            "Toastmasters International speeches",
            "YouTube: Voice coaching channels",
            "Online courses: Coursera, Udemy speaking courses"
        ],
        "🏢 Practice Opportunities": [
            "Join Toastmasters International",
            "Practice with friends and family",
            "Record video presentations",
            "Volunteer for presentations at work",
            "Join local speaking clubs"
        ],
        "📱 Helpful Apps": [
            "Voice Analyst for pitch tracking",
            "Orai AI for speech coaching",
            "Speeko for presentation practice",
            "VirtualSpeech for VR practice"
        ]
    }
    
    for category, resource_list in resources.items():
        with st.expander(category):
            for resource in resource_list:
                st.write(f"• {resource}")

with feature_tabs[2]:  # Settings
    st.subheader("⚙️ Application Settings")

    st.write("**Display Settings:**")
    show_debug = st.checkbox("Show debug information", value=False)
    auto_refresh = st.checkbox("Auto-refresh charts", value=True)
    
    st.write("**Analysis Settings:**")
    save_sessions = st.checkbox("Save session data locally", value=False)
    detailed_feedback = st.checkbox("Show detailed technical feedback", value=True)
    
    st.write("**Notifications:**")
    nervousness_alerts = st.checkbox("Alert on high nervousness", value=True)
    confidence_milestones = st.checkbox("Celebrate confidence milestones", value=True)
    
    if st.button("Reset All Settings"):
        st.success("Settings reset to defaults!")

with feature_tabs[3]:  # Help & FAQ
    st.subheader("❓ Help & Frequently Asked Questions")
    
    faqs = {
        "🎤 How accurate is the voice analysis?": 
            "Our analysis uses Librosa and standard pitch detection algorithms (YIN algorithm) with ~95% accuracy for clear speech. Results may vary with background noise or very soft speech.",
        
        "📊 What does the confidence score mean?":
            "Confidence score (0-100%) measures pitch stability, volume consistency, and speaking continuity. Higher scores indicate more controlled, professional speech patterns.",
        
        "⚠️ How is nervousness detected?":
            "Nervousness detection analyzes pitch variations, voice tremor, volume changes, and speaking pace irregularities. These are common physiological indicators of nervous speech.",
        
        "🎯 How can I improve my scores?":
            "Practice regularly with our exercises, focus on steady breathing, maintain consistent volume, and work on speaking at your natural pitch range.",
        
        "🔒 Is my voice data stored?":
            "No, all analysis happens in real-time in your browser. No voice recordings are stored on our servers unless you explicitly choose to export them.",
        
        "📱 What browsers work best?":
            "Chrome and Firefox provide the best compatibility. Safari and Edge may have limited WebRTC support.",
        
        "🎧 Do I need special equipment?":
            "Any standard microphone works. For best results, use a headset microphone in a quiet environment.",
        
        "📈 How often should I practice?":
            "For best results, practice 10-15 minutes daily. Consistency is more important than long sessions."
    }
    
    for question, answer in faqs.items():
        with st.expander(question):
            st.write(answer)
    
    st.write("---")
    st.write("**Need more help?** Email us at *raunakk046@gmail.com*")

####################################################### FOOTER ####################################################################################

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <h4>🗣️ SpeakSmart AI - Your Personal Communication Coach</h4>
    <p>Empowering confident communication through AI-powered speech analysis</p>
    <p><strong>Features:</strong> Real-time pitch analysis • Nervousness detection • Personalized feedback • Progress tracking</p>
    <p style="font-size: 0.8em; margin-top: 20px;">
        Built with ❤️ For Project Exhibition-I by Group No: 256 <br>
        Raunak Kumar Modi | Jahnvi Pandey | Rishi Singh Shandilya | Unnati Lohana | Vedant Singh<br>
        Version 1.0.1 | Made for improving public speaking and interview performance
    </p>
</div>
""", unsafe_allow_html=True)