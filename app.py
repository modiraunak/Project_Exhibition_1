# =============================================================================
# SpeakSmart AI — Communication Coach  v3.0  [FIXED]
# Group 256 | Project Exhibition-I
# Raunak Kumar Modi · Jahnvi Pandey · Rishi Singh Shandilya
# Unnati Lohana · Vedant Singh
#
# FIXES applied on top of v3.0
# ─────────────────────────────
# FIX 1 – BACKGROUND: CSS vars switched to a true dark theme (#0a0a0a base).
#          All tip/alert/good cards, grade card, and table colours updated for
#          dark-bg contrast.  Pill borders and muted text also adjusted.
#
# FIX 2 – REPORT ACCURACY: Introduced compute_overall() — a single source of
#          truth for the overall score.  Previously:
#            • PDF report used  conf - nerv*0.4  (no speaking-time bonus, no video)
#            • JSON export used  conf - nerv*0.4  (same problem)
#            • On-screen grade used the full formula but inline
#          Now all three call compute_overall() so PDF, JSON, and screen always
#          show the same number and letter grade.
#          The grade lookup table is also unified (was two different lists).
# =============================================================================

import io
import re
import time
from collections import deque
from datetime import datetime
import json

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import plotly.graph_objects as go

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

try:
    import mediapipe as mp
    import cv2
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import whisper as openai_whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS  — FIX 1: full dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpeakSmart AI",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    /* ── FIX 1: dark palette ── */
    --bg:      #0a0a0a;
    --card:    #141414;
    --ink:     #f0f0f0;
    --muted:   #9ca3af;
    --border:  #2a2a2a;
    --blue:    #3b82f6;
    --green:   #10b981;
    --amber:   #f59e0b;
    --red:     #ef4444;
    --mono:    'DM Mono', monospace;
}

* { font-family: 'DM Sans', sans-serif; box-sizing: border-box; }

/* page bg */
.stApp { background: var(--bg) !important; }

/* force Streamlit's inner containers to be transparent */
section[data-testid="stSidebar"] { background: #0f0f0f !important; border-right: 1px solid var(--border); }
.block-container { background: transparent !important; }

/* header */
.app-header {
    padding: 2rem 0 1.2rem;
    border-bottom: 1.5px solid var(--border);
    margin-bottom: 1.5rem;
}
.app-header h1 {
    font-size: 2.2rem; font-weight: 700; color: var(--ink);
    letter-spacing: -0.03em; margin: 0 0 .3rem;
}
.app-header h1 em { font-style: normal; color: var(--blue); }
.app-header p { color: var(--muted); font-weight: 300; margin: 0; font-size: .95rem; }

/* pill tags */
.pills { display: flex; flex-wrap: wrap; gap: 6px; margin: .8rem 0 1.2rem; }
.pill {
    border: 1.5px solid var(--border); border-radius: 99px;
    padding: 3px 12px; font-size: .73rem; font-weight: 500;
    color: var(--muted); background: var(--card); letter-spacing: .01em;
}

/* section label */
.slabel {
    font-family: var(--mono); font-size: .67rem; font-weight: 500;
    letter-spacing: .14em; text-transform: uppercase;
    color: var(--muted); margin: 0 0 .5rem;
}

/* tip / alert cards — FIX 1: dark-friendly versions */
.tip   { border-left: 3px solid var(--blue);  background: #1e2a3a; border-radius: 0 8px 8px 0; padding: .7rem 1rem; margin: .35rem 0; font-size: .86rem; color: #93c5fd; }
.alert { border-left: 3px solid var(--amber); background: #2a1f0a; border-radius: 0 8px 8px 0; padding: .7rem 1rem; margin: .35rem 0; font-size: .86rem; color: #fcd34d; }
.good  { border-left: 3px solid var(--green); background: #0a2a1a; border-radius: 0 8px 8px 0; padding: .7rem 1rem; margin: .35rem 0; font-size: .86rem; color: #6ee7b7; }

/* grade card — FIX 1: dark */
.grade-wrap { border: 2px solid var(--border); border-radius: 16px; padding: 2rem; text-align: center; background: var(--card); }
.grade-letter { font-size: 4rem; font-weight: 700; line-height: 1; color: var(--ink); }
.grade-score  { font-family: var(--mono); font-size: .95rem; color: var(--muted); margin: .4rem 0; }
.grade-msg    { font-size: .88rem; color: var(--muted); font-style: italic; }

/* filler word chip */
.filler-chip {
    display: inline-block; background: #3b1010; color: #fca5a5;
    border-radius: 99px; padding: 2px 10px; font-size: .75rem;
    font-weight: 600; margin: 3px; font-family: var(--mono);
}

hr.div { border: none; border-top: 1.5px solid var(--border); margin: 1.8rem 0; }

/* Streamlit widget label colour override */
label, .stSelectbox label, .stSlider label, .stCheckbox label {
    color: var(--muted) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
MAXLEN = 3000

def _fresh():
    return {
        # Audio
        "pitch_history":      deque(maxlen=MAXLEN),
        "rms_history":        deque(maxlen=MAXLEN),
        "timestamps":         deque(maxlen=MAXLEN),
        "confidence_history": deque(maxlen=MAXLEN),
        "current_confidence": 0,
        "nervousness_score":  0,
        "nervous_moments":    [],
        "coaching_tips":      [],
        # Filler words
        "filler_count":       {},
        "filler_rate":        0.0,
        "transcript_text":    "",
        # Video
        "eye_contact_score":   0,
        "posture_score":       0,
        "gesture_score":       0,
        "eye_contact_history": deque(maxlen=MAXLEN),
        "posture_history":     deque(maxlen=MAXLEN),
        # Session
        "session_duration":    0.0,
    }

for k, v in _fresh().items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="slabel">Mode</p>', unsafe_allow_html=True)
    mode = st.selectbox("Mode", [
        "Audio File Upload",
        "Video + Audio Upload",
        "Live Webcam (WebRTC)",
        "Live Mic (SoundDevice)",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<p class="slabel">Voice Settings</p>', unsafe_allow_html=True)
    fmin              = st.slider("Min Pitch Hz", 50, 200, 75, 5)
    fmax              = st.slider("Max Pitch Hz", 200, 500, 350, 10)
    silence_threshold = st.slider("Silence RMS threshold", 0.001, 0.05, 0.005, 0.001)
    analysis_window   = st.slider("Analysis window (s)", 3, 15, 8)

    st.markdown("---")
    st.markdown('<p class="slabel">Video Settings</p>', unsafe_allow_html=True)
    do_eye   = st.checkbox("Eye Contact", True)
    do_pose  = st.checkbox("Posture", True)
    do_hands = st.checkbox("Gestures", True)
    do_expr  = st.checkbox("Expressions", True)
    vid_wt   = st.slider("Video weight in grade (%)", 0, 100, 40)

    st.markdown("---")
    st.markdown('<p class="slabel">Nervousness</p>', unsafe_allow_html=True)
    sensitivity = st.select_slider("Sensitivity",
                                   options=["Low","Medium","High"], value="Medium")

    st.markdown("---")
    if st.button("↺ Reset session", use_container_width=True):
        for k, v in _fresh().items():
            st.session_state[k] = v
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🗣️ Speak<em>Smart</em> AI</h1>
  <p>Voice · Posture · Eye contact · Filler words · Personalised coaching</p>
</div>
<div class="pills">
  <span class="pill">🎙️ Pitch Analysis</span>
  <span class="pill">📹 Body Language</span>
  <span class="pill">👁️ Eye Contact</span>
  <span class="pill">🧍 Posture</span>
  <span class="pill">🤚 Gestures</span>
  <span class="pill">🔤 Filler Words</span>
  <span class="pill">⚠️ Nervousness</span>
  <span class="pill">📄 PDF Report</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 – UNIFIED SCORING  (single source of truth used by UI, PDF, and JSON)
# ─────────────────────────────────────────────────────────────────────────────

# Unified grade table — descending order, first match wins
GRADE_MAP = [
    (85, "A+", "Outstanding communication skills!"),
    (75, "A",  "Excellent — strong confidence and presence."),
    (65, "B+", "Good — minor refinements will push you to excellent."),
    (55, "B",  "Solid — focus on eye contact and pitch consistency."),
    (45, "C+", "Fair — work on nervousness and posture."),
    (35, "C",  "Average — significant improvement with practice."),
    ( 0, "D",  "Keep going — every expert was once a beginner."),
]


def compute_overall(
    conf: int,
    nerv: int,
    pitch_history,
    timestamps,
    eye: int = 0,
    pos: int = 0,
    gest: int = 0,
    has_video: bool = False,
    video_weight_pct: int = 40,
) -> tuple[float, str, str]:
    """
    Single source of truth for overall score + grade.

    Audio score = max(0, confidence - nervousness*0.4 + speaking_time_bonus)
      • speaking_time_bonus: up to +10 pts for covering ≥ 80 % of recording
    Video score = eye*0.40 + posture*0.40 + gesture*0.20
    Overall = audio*(1-w) + video*w   [w = video_weight_pct/100, only if has_video]

    Returns (overall_float, letter_grade, grade_message).
    """
    vp  = [p for p in pitch_history if p is not None and np.isfinite(p)]
    tt  = max(list(timestamps), default=1)
    sp_t = len(vp) * SpeechAnalyzer.HOP
    bonus = min(sp_t / max(tt, 1) * 20, 10)           # up to +10 pts

    audio_score = float(np.clip(conf - nerv * 0.4 + bonus, 0, 100))

    if has_video:
        video_score = eye * 0.40 + pos * 0.40 + gest * 0.20
        w           = video_weight_pct / 100
        overall     = audio_score * (1 - w) + video_score * w
    else:
        overall = audio_score

    overall = float(np.clip(overall, 0, 100))
    letter, msg = next((g, m) for t, g, m in GRADE_MAP if overall >= t)
    return overall, letter, msg


# ─────────────────────────────────────────────────────────────────────────────
# FILLER WORD DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
FILLERS = [
    "um", "uh", "er", "ah", "like", "you know", "you know what i mean",
    "basically", "literally", "actually", "right", "so", "i mean",
    "kind of", "sort of", "just", "okay so", "and uh", "and um",
]

def detect_fillers(text: str, duration_s: float) -> tuple[dict, float]:
    text_lower = text.lower()
    counts: dict[str, int] = {}
    for filler in sorted(FILLERS, key=len, reverse=True):
        pattern = r'\b' + re.escape(filler) + r'\b'
        found   = re.findall(pattern, text_lower)
        if found:
            counts[filler] = len(found)
            text_lower = re.sub(pattern, ' ', text_lower)
    total = sum(counts.values())
    rate  = (total / max(duration_s, 1)) * 60
    return counts, rate


def transcribe_audio(audio: np.ndarray, sr: int) -> str:
    if not WHISPER_AVAILABLE:
        return ""
    try:
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        model     = openai_whisper.load_model("tiny")
        result    = model.transcribe(audio_16k, language="en", fp16=False)
        return result.get("text", "")
    except Exception:
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# SPEECH ANALYZER
# ─────────────────────────────────────────────────────────────────────────────
class SpeechAnalyzer:
    CHUNK = 0.25
    HOP   = 0.05

    def pitch_rms(self, chunk: np.ndarray, sr: int) -> tuple:
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < silence_threshold:
            return np.nan, rms
        if len(chunk) < int(3.0 / fmin * sr):
            return np.nan, rms
        try:
            hop = max(64, len(chunk) // 16)
            f0  = librosa.yin(
                chunk.astype(np.float32),
                fmin=fmin, fmax=fmax, sr=sr,
                hop_length=hop,
                frame_length=min(2048, len(chunk) // 2),
            )
            valid = f0[(np.isfinite(f0)) & (f0 > fmin * 0.9) & (f0 < fmax * 1.1)]
            if len(valid) < 2:
                return np.nan, rms
            return float(np.median(valid)), rms
        except Exception:
            return np.nan, rms

    def confidence(self, pitches: list, rms_list: list, win_s: float) -> int:
        n = max(10, int(win_s / self.HOP))
        s = max(0, len(pitches) - n)
        vp, vr = [], []
        for i in range(s, len(pitches)):
            p = pitches[i]
            r = rms_list[i] if i < len(rms_list) else 0.0
            if p is not None and np.isfinite(p) and r >= silence_threshold:
                vp.append(p); vr.append(r)
        if len(vp) < 5:
            return 0
        pa, ra = np.array(vp), np.array(vr)
        cv_p  = pa.std() / pa.mean() if pa.mean() > 0 else 1.0
        stab  = max(0.0, 1.0 - cv_p / 0.30)
        cv_r  = ra.std() / ra.mean() if ra.mean() > 0 else 1.0
        vol   = max(0.0, 1.0 - cv_r / 0.70)
        cont  = min(1.0, len(vp) / max(n * 0.7, 1))
        ratio = (pa.mean() - fmin) / max(fmax - fmin, 1)
        rng   = max(0.0, min(1.0, 1.0 - abs(ratio - 0.45) * 2))
        return int(np.clip(100 * (stab*0.35 + vol*0.25 + cont*0.25 + rng*0.15), 0, 100))

    def nervousness(self, pitches: list, rms_list: list, win_s: float) -> tuple:
        mult   = {"Low": 0.6, "Medium": 1.0, "High": 1.5}[sensitivity]
        voiced = [p for p in pitches if p is not None and np.isfinite(p)]
        if len(voiced) < 8:
            return 0, []
        pa    = np.array(voiced)
        score = 0.0; notes = []
        cv = pa.std() / pa.mean() if pa.mean() > 0 else 0
        if cv > 0.20:
            score += min(30, cv * 120) * mult
            notes.append(f"High pitch variability (CV={cv:.2f})")
        if len(voiced) > 5:
            jitter = np.mean(np.abs(np.diff(pa))) / pa.mean()
            if jitter > 0.04:
                score += min(25, jitter * 400) * mult
                notes.append("Voice tremor / jitter")
        baseline = np.percentile(pa, 40)
        recent   = np.mean(pa[-max(3, len(pa)//4):])
        if baseline > 0 and (recent - baseline) / baseline > 0.12:
            score += 15 * mult
            notes.append("Pitch rising toward end of speech")
        vr = [r for r in rms_list if r >= silence_threshold]
        if len(vr) >= 5:
            ra   = np.array(vr)
            cv_r = ra.std() / ra.mean() if ra.mean() > 0 else 0
            if cv_r > 0.50:
                score += 10 * mult
                notes.append("Inconsistent volume")
        tot = len(rms_list)
        sil = sum(1 for r in rms_list if r < silence_threshold)
        if tot > 0 and sil / tot > 0.50:
            score += 15 * mult
            notes.append(f"Long pauses ({sil/tot*100:.0f}% silence)")
        return int(min(score, 100)), notes

    def coaching_tips(self, conf, nerv, notes,
                      eye=None, pos=None, gest=None,
                      filler_rate=0.0) -> list:
        tips = []
        if conf < 30:
            tips += ["🎯 Practise sustained vowel sounds to build tonal steadiness",
                     "😮‍💨 Do 2 min of diaphragmatic breathing before speaking"]
        elif conf < 55:
            tips += ["✅ Good start — focus on keeping volume consistent",
                     "⏱️ Slow down slightly; rushing reduces perceived confidence"]
        elif conf < 75:
            tips.append("💪 Strong voice! Add pitch variety to avoid monotone delivery")
        if nerv > 65:
            tips += ["😰 High tension — breathe in for 4, out for 6 between sentences",
                     "🎭 Visualise a successful outcome before you begin"]
        elif nerv > 35:
            tips.append("⚠️ Mild tension — relax your jaw and shoulders consciously")
        joined = " ".join(notes).lower()
        if "variability"  in joined: tips.append("📊 Read aloud daily at a slow, measured pace")
        if "tremor"       in joined: tips.append("🤲 Lip trills and humming exercises reduce vocal tremor")
        if "rising"       in joined: tips.append("⬇️ Speak from your chest — it naturally lowers pitch")
        if "volume"       in joined: tips.append("🔊 Record yourself and listen back for volume dips")
        if "pause"        in joined: tips.append("⏰ Replace filler pauses with one deliberate silent beat")
        if filler_rate > 10:
            tips.append(f"🔤 You use ~{filler_rate:.0f} filler words/min — try pausing silently instead")
        elif filler_rate > 5:
            tips.append(f"🔤 ~{filler_rate:.0f} fillers/min detected — awareness is the first step")
        if eye  is not None and eye  < 50: tips.append("👁️ Look at the camera LENS — put a dot above it as a guide")
        if eye  is not None and eye  < 70: tips.append("👁️ Aim for 70-80% camera time; brief glances are fine")
        if pos  is not None and pos  < 50: tips.append("🧍 Ears-over-shoulders-over-hips: the posture check")
        if gest is not None and gest < 35: tips.append("🤚 Open-palm gestures project confidence and openness")
        return list(dict.fromkeys(tips))


analyzer = SpeechAnalyzer()

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO ANALYZER
# ─────────────────────────────────────────────────────────────────────────────
class VideoAnalyzer:
    def __init__(self):
        self.ready     = MEDIAPIPE_AVAILABLE
        self.face_mesh = self.pose = self.hands = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_fm   = mp.solutions.face_mesh
                self.mp_pose = mp.solutions.pose
                self.mp_hand = mp.solutions.hands
                self.mp_draw = mp.solutions.drawing_utils
                self.mp_ds   = mp.solutions.drawing_styles
            except Exception:
                self.ready = False

    def open(self):
        if not self.ready: return False
        try:
            if do_eye or do_expr:
                self.face_mesh = self.mp_fm.FaceMesh(
                    max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
            if do_pose:
                self.pose = self.mp_pose.Pose(
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
            if do_hands:
                self.hands = self.mp_hand.Hands(
                    max_num_hands=2,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
            return True
        except Exception:
            return False

    def close(self):
        for m in (self.face_mesh, self.pose, self.hands):
            if m:
                try: m.close()
                except: pass

    def _eye_contact(self, face_lm) -> float:
        lm = face_lm.landmark
        try:
            li, ri   = lm[468], lm[473]
            lo, ro   = lm[33],  lm[263]
            li_in    = lm[133]
            ri_in    = lm[362]
            iod = np.hypot(ro.x - lo.x, ro.y - lo.y)
            if iod < 1e-4: return 50.0
            l_cx = (lo.x + li_in.x) / 2;  l_cy = (lo.y + li_in.y) / 2
            r_cx = (ro.x + ri_in.x) / 2;  r_cy = (ro.y + ri_in.y) / 2
            l_off = np.hypot(li.x - l_cx, li.y - l_cy) / iod
            r_off = np.hypot(ri.x - r_cx, ri.y - r_cy) / iod
            return float(np.clip(100 - (l_off + r_off) / 2 * 350, 0, 100))
        except Exception:
            return 50.0

    def _posture(self, pose_lm) -> tuple:
        if not pose_lm: return 50.0, []
        lm = pose_lm.landmark
        P  = self.mp_pose.PoseLandmark
        issues, score = [], 100
        try:
            ls, rs = lm[P.LEFT_SHOULDER],  lm[P.RIGHT_SHOULDER]
            le, re = lm[P.LEFT_EAR],       lm[P.RIGHT_EAR]
            sw = np.hypot(rs.x - ls.x, rs.y - ls.y)
            if sw < 1e-3: return 50.0, []
            if abs(ls.y - rs.y) / sw > 0.12:
                score -= 20; issues.append("Uneven shoulders")
            if abs((le.x + re.x)/2 - (ls.x + rs.x)/2) / sw > 0.35:
                score -= 15; issues.append("Head tilted sideways")
            ear_rise = (ls.y + rs.y)/2 - (le.y + re.y)/2
            if ear_rise < sw * 0.30:
                score -= 15; issues.append("Possible slouching")
            vis = min(ls.visibility, rs.visibility, le.visibility, re.visibility)
            if vis < 0.5:
                score = int(score * vis * 2)
            return max(0, min(100, score)), issues
        except Exception:
            return 50.0, []

    def _gestures(self, hand_lms, pose_lm) -> float:
        wrists_visible = False
        if pose_lm and do_pose:
            try:
                P  = self.mp_pose.PoseLandmark
                lm = pose_lm.landmark
                wrists_visible = (lm[P.LEFT_WRIST].visibility > 0.5
                                  or lm[P.RIGHT_WRIST].visibility > 0.5)
            except Exception:
                pass
        if not hand_lms:
            return 30.0 if wrists_visible else 50.0
        total = 0.0
        for hand in hand_lms:
            lm    = hand.landmark
            wrist = lm[0]
            tips  = [lm[4], lm[8], lm[12], lm[16], lm[20]]
            palm  = np.hypot(lm[9].x - wrist.x, lm[9].y - wrist.y)
            if palm < 1e-4: continue
            avg_ext = np.mean([np.hypot(t.x-wrist.x, t.y-wrist.y)/palm for t in tips])
            total += float(np.clip((avg_ext - 1.0) / 1.5, 0, 1)) * 100
        return min(100.0, total / len(hand_lms))

    def _expression(self, face_lm) -> tuple:
        if not face_lm: return "neutral 😐", 50.0
        lm = face_lm.landmark
        try:
            lc, rc = lm[61], lm[291]; tm, bm = lm[13], lm[14]
            sr = np.hypot(rc.x-lc.x, rc.y-lc.y) / max(np.hypot(bm.x-tm.x, bm.y-tm.y), 1e-4)
            if sr > 4.5: return "smiling 😊", min(100.0, sr*12)
            elif sr > 2.5: return "neutral 😐", 55.0
            else: return "tense 😟", 25.0
        except Exception:
            return "neutral 😐", 50.0

    def _hud(self, frame, res):
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (8, 8), (250, 108), (10, 10, 20), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        fn, y = cv2.FONT_HERSHEY_SIMPLEX, 26
        if res["face"]:
            ec  = res["eye"]
            col = (0,200,80) if ec>65 else (0,165,255) if ec>40 else (60,60,220)
            cv2.putText(frame, f"Eye  {ec:.0f}%",  (14,y), fn, 0.5, col, 1); y+=22
        p   = res["posture"]
        pc  = (0,200,80) if p>70 else (0,165,255) if p>45 else (60,60,220)
        cv2.putText(frame, f"Post {p:.0f}%",  (14,y), fn, 0.5, pc, 1); y+=22
        cv2.putText(frame, res["expr"],        (14,y), fn, 0.45, (200,200,200), 1); y+=22
        cv2.putText(frame, f"Hands {res['hands']}", (14,y), fn, 0.45, (150,150,150), 1)
        cv2.circle(frame, (w-16, 16), 6, (60,60,220), -1)
        cv2.putText(frame, "LIVE", (w-56,21), fn, 0.42, (255,255,255), 1)

    def process(self, bgr) -> tuple:
        if not self.ready: return self._empty(), bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self._empty()
        face_lm = pose_lm = hand_lms = None
        if self.face_mesh and (do_eye or do_expr):
            fr = self.face_mesh.process(rgb)
            if fr.multi_face_landmarks:
                face_lm    = fr.multi_face_landmarks[0]
                res["face"] = True
                self.mp_draw.draw_landmarks(
                    bgr, face_lm, self.mp_fm.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_ds.get_default_face_mesh_contours_style())
                if do_eye:  res["eye"]  = self._eye_contact(face_lm)
                if do_expr: res["expr"], res["smile"] = self._expression(face_lm)
        if self.pose and do_pose:
            pr = self.pose.process(rgb)
            if pr.pose_landmarks:
                pose_lm = pr.pose_landmarks
                res["posture"], res["issues"] = self._posture(pose_lm)
                self.mp_draw.draw_landmarks(
                    bgr, pose_lm, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_ds.get_default_pose_landmarks_style())
        if self.hands and do_hands:
            hr = self.hands.process(rgb)
            if hr.multi_hand_landmarks:
                hand_lms      = hr.multi_hand_landmarks
                res["hands"]  = len(hand_lms)
                for h in hand_lms:
                    self.mp_draw.draw_landmarks(
                        bgr, h, self.mp_hand.HAND_CONNECTIONS,
                        self.mp_ds.get_default_hand_landmarks_style(),
                        self.mp_ds.get_default_hand_connections_style())
            res["gesture"] = self._gestures(hand_lms, pose_lm)
        self._hud(bgr, res)
        return res, bgr

    def _empty(self):
        return {"face":False,"eye":50.0,"posture":50.0,"issues":[],
                "gesture":50.0,"expr":"—","smile":50.0,"hands":0}


video_analyzer = VideoAnalyzer()

# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT GENERATOR  — FIX 2: uses compute_overall() for accurate scores
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_report() -> bytes:
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles  import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units   import cm
    from reportlab.lib         import colors

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    h1 = ParagraphStyle("H1", parent=styles["Heading1"],
                         fontSize=20, spaceAfter=4, textColor=colors.HexColor("#1a1a1a"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                         fontSize=13, spaceBefore=14, spaceAfter=4,
                         textColor=colors.HexColor("#2563eb"))
    body = ParagraphStyle("Body", parent=styles["Normal"],
                          fontSize=10, leading=15, textColor=colors.HexColor("#374151"))
    mono = ParagraphStyle("Mono", parent=styles["Normal"],
                          fontSize=9, fontName="Courier",
                          textColor=colors.HexColor("#6b7280"))

    story.append(Paragraph("SpeakSmart AI — Session Report", h1))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}  |  "
        f"Version 3.0  |  Group 256", mono))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#e4e2dc"), spaceAfter=10))

    # ── FIX 2: pull scores from session state, compute via unified function ──
    conf = st.session_state.current_confidence
    nerv = st.session_state.nervousness_score
    ec   = st.session_state.eye_contact_score
    pos  = st.session_state.posture_score
    gest = st.session_state.gesture_score
    fr   = st.session_state.filler_rate
    has_video = ec > 0 or pos > 0

    overall, grade, grade_msg = compute_overall(
        conf=conf, nerv=nerv,
        pitch_history=list(st.session_state.pitch_history),
        timestamps=list(st.session_state.timestamps),
        eye=ec, pos=pos, gest=gest,
        has_video=has_video,
        video_weight_pct=vid_wt,
    )

    story.append(Paragraph("Score Summary", h2))
    table_data = [
        ["Metric", "Score", "Status"],
        ["Voice Confidence",  f"{conf}%",   "Good" if conf >= 55 else "Needs work"],
        ["Nervousness",       f"{nerv}%",   "Good" if nerv <= 35 else "High"],
        ["Filler Words/min",  f"{fr:.1f}",  "Good" if fr <= 5 else "Reduce"],
    ]
    if has_video:
        table_data += [
            ["Eye Contact",  f"{ec}%",   "Good" if ec >= 60 else "Improve"],
            ["Posture",      f"{pos}%",  "Good" if pos >= 60 else "Improve"],
            ["Gestures",     f"{gest}%", "Good" if gest >= 50 else "Improve"],
        ]
    table_data.append(["Overall Score", f"{overall:.1f}%", f"Grade: {grade}"])

    t = Table(table_data, colWidths=[7*cm, 3*cm, 5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#2563eb")),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0),  10),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#f7f6f3")]),
        ("FONTSIZE",      (0,1), (-1,-1), 9),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#e4e2dc")),
        ("FONTNAME",      (0,-1),(-1,-1), "Helvetica-Bold"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Paragraph(f"Grade message: {grade_msg}", body))
    story.append(Spacer(1, 12))

    nm = st.session_state.nervous_moments
    if nm:
        story.append(Paragraph("Nervousness Episodes", h2))
        deduped = []
        for m in nm:
            if not deduped or m["time"] - deduped[-1]["time"] > 2.0:
                deduped.append(m)
        for i, m in enumerate(deduped[:8], 1):
            story.append(Paragraph(
                f"<b>#{i}</b>  at {m['time']:.1f}s — score {m['score']}%", body))
            for note in set(m.get("notes", [])):
                story.append(Paragraph(f"&nbsp;&nbsp;• {note}", mono))
        story.append(Spacer(1, 6))

    fc = st.session_state.filler_count
    if fc:
        story.append(Paragraph("Filler Words Detected", h2))
        fw_rows = [["Filler", "Count"]] + sorted(
            [[k, str(v)] for k, v in fc.items()], key=lambda x: -int(x[1]))
        ft = Table(fw_rows, colWidths=[8*cm, 7*cm])
        ft.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#fee2e2")),
            ("TEXTCOLOR",     (0,0), (-1,0),  colors.HexColor("#991b1b")),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#fecaca")),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#fff5f5")]),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(ft)
        story.append(Spacer(1, 6))

    tips = st.session_state.coaching_tips
    if tips:
        story.append(Paragraph("Personalised Coaching Tips", h2))
        for tip in tips:
            clean = re.sub(r'[^\x00-\x7F]+', '', tip).strip(" –-")
            story.append(Paragraph(f"• {clean}", body))
        story.append(Spacer(1, 6))

    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#e4e2dc"), spaceBefore=10))
    story.append(Paragraph(
        "SpeakSmart AI v3.0  |  Group 256  |  Project Exhibition-I  |  "
        "Raunak Kumar Modi, Jahnvi Pandey, Rishi Singh Shandilya, "
        "Unnati Lohana, Vedant Singh", mono))

    doc.build(story)
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def _clear_audio():
    for k in ("pitch_history","rms_history","timestamps","confidence_history"):
        st.session_state[k] = deque(maxlen=MAXLEN)
    st.session_state["nervous_moments"] = []


def _run_audio(audio: np.ndarray, sr: int):
    hop_n   = int(sr * SpeechAnalyzer.HOP)
    chunk_n = int(sr * SpeechAnalyzer.CHUNK)
    n_hops  = max(1, (len(audio) - chunk_n) // hop_n)
    every   = int(1.0 / SpeechAnalyzer.HOP)
    prog    = st.progress(0)

    for idx, start in enumerate(range(0, len(audio) - chunk_n, hop_n)):
        chunk      = audio[start : start + chunk_n]
        pitch, rms = analyzer.pitch_rms(chunk, sr)
        t          = start / sr

        st.session_state.pitch_history.append(pitch if np.isfinite(pitch) else None)
        st.session_state.rms_history.append(rms)
        st.session_state.timestamps.append(t)
        prog.progress(min(1.0, idx / n_hops))

        if idx % every == 0:
            ph = list(st.session_state.pitch_history)
            rh = list(st.session_state.rms_history)
            c  = analyzer.confidence(ph, rh, analysis_window)
            n, notes = analyzer.nervousness(ph, rh, analysis_window)
            st.session_state.current_confidence = c
            st.session_state.nervousness_score  = n
            st.session_state.confidence_history.append(c)
            if n > 55 and t > 0:
                st.session_state.nervous_moments.append(
                    {"time": t, "score": n, "notes": notes})

    prog.progress(1.0)


def process_audio(uploaded) -> bool:
    try:
        audio, sr = librosa.load(uploaded, sr=None, mono=True)
        st.success(f"✅ Loaded — {len(audio)/sr:.1f}s @ {sr} Hz")
    except Exception as e:
        st.error(f"Could not load audio: {e}"); return False

    _clear_audio()
    _run_audio(audio, sr)

    with st.spinner("🔤 Detecting filler words…"):
        transcript = transcribe_audio(audio, sr)
        st.session_state.transcript_text = transcript
        dur = len(audio) / sr
        fc, fr = detect_fillers(transcript, dur)
        st.session_state.filler_count = fc
        st.session_state.filler_rate  = fr

    all_notes = [n for m in st.session_state.nervous_moments for n in m["notes"]]
    st.session_state.coaching_tips = analyzer.coaching_tips(
        st.session_state.current_confidence,
        st.session_state.nervousness_score,
        all_notes,
        filler_rate=fr,
    )
    return True


def process_video(uploaded) -> bool:
    if not MEDIAPIPE_AVAILABLE:
        st.warning("MediaPipe not found — audio-only analysis.")
        return process_audio(uploaded)

    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read()); tmp_path = tmp.name

    try:
        try:
            audio, sr = librosa.load(tmp_path, sr=None, mono=True)
            audio_ok  = True
            st.info(f"🎵 Audio: {len(audio)/sr:.1f}s @ {sr} Hz")
        except Exception as e:
            st.warning(f"Audio extraction skipped: {e}"); audio_ok = False

        cap   = cv2.VideoCapture(tmp_path)
        n_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        st.success(f"✅ Video: {n_frm/fps:.1f}s — {n_frm} frames @ {fps:.0f} fps")

        video_analyzer.open()
        _clear_audio()
        for k in ("eye_contact_history","posture_history"):
            st.session_state[k] = deque(maxlen=MAXLEN)

        if audio_ok:
            _run_audio(audio, sr)
            ph_l = list(st.session_state.pitch_history)
            rh_l = list(st.session_state.rms_history)
            step = int(1.0 / SpeechAnalyzer.HOP)
            win  = int(analysis_window / SpeechAnalyzer.HOP)
            st.session_state.confidence_history = deque(maxlen=MAXLEN)
            for i in range(win, len(ph_l), step):
                c = analyzer.confidence(ph_l[:i], rh_l[:i], analysis_window)
                n, notes = analyzer.nervousness(ph_l[:i], rh_l[:i], analysis_window)
                st.session_state.confidence_history.append(c)
                st.session_state.current_confidence = c
                st.session_state.nervousness_score  = n
                if n > 55:
                    st.session_state.nervous_moments.append(
                        {"time": i*SpeechAnalyzer.HOP, "score": n, "notes": notes})

        with st.spinner("🔤 Detecting filler words…"):
            transcript = transcribe_audio(audio, sr) if audio_ok else ""
            st.session_state.transcript_text = transcript
            dur = (len(audio)/sr) if audio_ok else 0
            fc, fr = detect_fillers(transcript, dur)
            st.session_state.filler_count = fc
            st.session_state.filler_rate  = fr

        st.info("📹 Analysing body language…")
        prog  = st.progress(0); vph = st.empty()
        every = max(1, int(fps / 5))
        ec_buf, p_buf, g_buf = [], [], []
        fc_cnt = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            fc_cnt += 1
            if fc_cnt % every == 0:
                res, ann = video_analyzer.process(frame)
                ec_buf.append(res["eye"]); p_buf.append(res["posture"])
                g_buf.append(res["gesture"])
                st.session_state.eye_contact_history.append(res["eye"])
                st.session_state.posture_history.append(res["posture"])
                vph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
            prog.progress(min(1.0, fc_cnt / n_frm))

        cap.release(); prog.progress(1.0); vph.empty()

        if ec_buf: st.session_state.eye_contact_score = int(np.mean(ec_buf))
        if p_buf:  st.session_state.posture_score     = int(np.mean(p_buf))
        if g_buf:  st.session_state.gesture_score     = int(np.mean(g_buf))

        all_notes = [n for m in st.session_state.nervous_moments for n in m["notes"]]
        st.session_state.coaching_tips = analyzer.coaching_tips(
            st.session_state.current_confidence,
            st.session_state.nervousness_score,
            all_notes,
            eye=st.session_state.eye_contact_score,
            pos=st.session_state.posture_score,
            gest=st.session_state.gesture_score,
            filler_rate=fr,
        )
        video_analyzer.close()
        os.unlink(tmp_path)
        return True

    except Exception as e:
        import traceback
        st.error(f"Error: {e}"); st.code(traceback.format_exc())
        try: os.unlink(tmp_path)
        except: pass
        return False

# ─────────────────────────────────────────────────────────────────────────────
# WEBRTC LIVE PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
class LiveVideoProcessor:
    def __init__(self):
        self.va  = VideoAnalyzer(); self.va.open(); self.cnt = 0
    def recv(self, frame):
        bgr = frame.to_ndarray(format="bgr24"); self.cnt += 1
        if self.cnt % 5 == 0:
            try:
                res, ann = self.va.process(bgr)
                st.session_state.eye_contact_score = int(res["eye"])
                st.session_state.posture_score     = int(res["posture"])
                st.session_state.gesture_score     = int(res["gesture"])
                return av.VideoFrame.from_ndarray(ann, format="bgr24")
            except: pass
        return frame

# ─────────────────────────────────────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="slabel">Input</p>', unsafe_allow_html=True)

if mode == "Audio File Upload":
    uf = st.file_uploader("Upload audio (WAV / MP3 / M4A / FLAC)",
                          type=["wav","mp3","m4a","flac","aac"])
    if uf and st.button("Analyse →", type="primary"):
        if process_audio(uf):
            st.success("Done — results below.")

elif mode == "Video + Audio Upload":
    if not MEDIAPIPE_AVAILABLE:
        st.error("Install: pip install mediapipe opencv-python-headless")
    uv = st.file_uploader("Upload video (MP4 / MOV / AVI / MKV)",
                          type=["mp4","mov","avi","mkv","webm"])
    if uv:
        st.video(uv)
        if st.button("Analyse Video + Audio →", type="primary"):
            with st.status("Running analysis…", expanded=True) as s:
                uv.seek(0); ok = process_video(uv)
                s.update(label="✅ Done!" if ok else "❌ Failed",
                         state="complete" if ok else "error")

elif mode == "Live Webcam (WebRTC)":
    if not WEBRTC_AVAILABLE:
        st.error("Install: pip install streamlit-webrtc aiortc")
    elif not MEDIAPIPE_AVAILABLE:
        st.error("Install: pip install mediapipe opencv-python-headless")
    else:
        cam_col, stat_col = st.columns([2, 1])
        with cam_col:
            ctx = webrtc_streamer(
                key="ss",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration(
                    {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}),
                video_processor_factory=LiveVideoProcessor,
                media_stream_constraints={"video":True,"audio":False},
                async_processing=True,
            )
        with stat_col:
            st.markdown('<p class="slabel">Live Body Language</p>', unsafe_allow_html=True)
            m_ec, m_po, m_ge, tip_ph = st.empty(),st.empty(),st.empty(),st.empty()
            if ctx and ctx.state.playing:
                ec = st.session_state.eye_contact_score
                p  = st.session_state.posture_score
                g  = st.session_state.gesture_score
                m_ec.metric("👁️ Eye Contact", f"{ec}%", "Good" if ec>65 else "Improve")
                m_po.metric("🧍 Posture",     f"{p}%",  "Good" if p>70  else "Adjust")
                m_ge.metric("🤚 Gestures",    f"{g}%",  "Natural" if g>50 else "More")
                if   ec<50: tip_ph.warning("Look directly at the camera lens")
                elif p<50:  tip_ph.warning("Straighten your posture")
                else:       tip_ph.success("✅ Looking great!")
            else:
                m_ec.metric("👁️ Eye Contact","—")
                m_po.metric("🧍 Posture","—")
                m_ge.metric("🤚 Gestures","—")
                tip_ph.info("Start camera to see live metrics")

elif mode == "Live Mic (SoundDevice)":
    if not SOUNDDEVICE_AVAILABLE:
        st.error("Install: pip install sounddevice")
    else:
        dur_s = st.slider("Duration (s)", 5, 60, 15)
        if st.button("🎤 Record", type="primary"):
            try:
                import soundfile as sf
                chunks, prog, stxt = [], st.progress(0), st.empty()
                def _cb(indata, frames, ti, stat): chunks.append(indata.copy())
                with sd.InputStream(callback=_cb, channels=1,
                                    samplerate=44100, dtype=np.float32):
                    for i in range(dur_s*10):
                        time.sleep(0.1)
                        prog.progress((i+1)/(dur_s*10))
                        stxt.text(f"Recording… {dur_s-i//10}s left")
                if chunks:
                    full = np.concatenate(chunks).flatten()
                    buf  = io.BytesIO()
                    sf.write(buf, full, 44100, format="WAV"); buf.seek(0)
                    process_audio(buf)
            except Exception as e:
                st.error(f"Recording error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
has_audio = len(st.session_state.timestamps) > 0
has_video = st.session_state.eye_contact_score > 0 or st.session_state.posture_score > 0

if not (has_audio or has_video):
    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown('<p class="slabel">Practice Scenarios</p>', unsafe_allow_html=True)
    scenarios = {
        "Job Interview":  ["Tell me about yourself","Describe a challenge you overcame",
                           "Why do you want this role?"],
        "Presentation":   ["Introduce your project","Pitch a new idea",
                           "Deliver a motivational opening"],
        "Networking":     ["Introduce yourself in 60 seconds","Make small talk"],
        "Academic":       ["Explain a concept clearly","Defend your thesis"],
    }
    sel = st.selectbox("Choose a scenario:", list(scenarios.keys()))
    if sel:
        with st.expander(f"Prompts — {sel}", expanded=True):
            for i, p in enumerate(scenarios[sel], 1):
                st.write(f"**{i}.** {p}")
        st.info("Record yourself answering, then upload for analysis.")

else:
    conf = st.session_state.current_confidence
    nerv = st.session_state.nervousness_score

    # ── KEY METRICS ───────────────────────────────────────────────────────────
    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown('<p class="slabel">Key Metrics</p>', unsafe_allow_html=True)

    n_cols = 7 if (has_audio and has_video) else 5 if has_audio else 3
    cols   = st.columns(n_cols)

    def _icon(v, lo, hi): return "🟢" if v>=hi else "🟡" if v>=lo else "🔴"

    cols[0].metric("🎤 Voice Confidence",
                   f"{_icon(conf,50,70)} {conf}%")
    cols[1].metric("⚠️ Nervousness",
                   f"{'🟢' if nerv<35 else '🟡' if nerv<65 else '🔴'} {nerv}%")

    if has_audio:
        vp  = [p for p in st.session_state.pitch_history if p is not None and np.isfinite(p)]
        tt  = max(list(st.session_state.timestamps), default=1)
        cov = len(vp) * SpeechAnalyzer.HOP / tt * 100
        cols[2].metric("🎙️ Speaking Coverage", f"{cov:.0f}%")
        cols[3].metric("🚨 Nervous Episodes",
                       str(len(st.session_state.nervous_moments)))
        fr = st.session_state.filler_rate
        fr_icon = "🟢" if fr<=5 else "🟡" if fr<=10 else "🔴"
        cols[4].metric("🔤 Fillers/min", f"{fr_icon} {fr:.1f}")

    if has_video:
        base = 5 if has_audio else 0
        ec   = st.session_state.eye_contact_score
        pos  = st.session_state.posture_score
        cols[base].metric("👁️ Eye Contact",
                          f"{_icon(ec,50,70)} {ec}%")
        cols[base+1].metric("🧍 Posture",
                            f"{_icon(pos,50,70)} {pos}%")

    # ── FILLER WORDS ──────────────────────────────────────────────────────────
    fc = st.session_state.filler_count
    if fc:
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        st.markdown('<p class="slabel">Filler Words</p>', unsafe_allow_html=True)
        fr = st.session_state.filler_rate
        chips = "".join(
            f'<span class="filler-chip">{w} ×{c}</span>'
            for w, c in sorted(fc.items(), key=lambda x: -x[1])
        )
        st.markdown(chips, unsafe_allow_html=True)
        if fr > 10:
            st.markdown(
                f'<div class="alert">⚠️ <strong>{fr:.0f} filler words/min</strong> — '
                'replace them with a confident 1-second pause.</div>',
                unsafe_allow_html=True)
        elif fr > 5:
            st.markdown(
                f'<div class="alert">ℹ️ <strong>{fr:.1f} fillers/min</strong> — '
                'you are aware of them, which is the first step.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="good">✅ Low filler rate ({fr:.1f}/min) — '
                'great use of deliberate pauses.</div>',
                unsafe_allow_html=True)

        if not WHISPER_AVAILABLE:
            st.info("ℹ️ Install `openai-whisper` to enable transcript-based filler detection.")

    elif not WHISPER_AVAILABLE and has_audio:
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        st.info("🔤 **Filler word detection** requires `openai-whisper`. "
                "Install with: `pip install openai-whisper`")

    # ── VIDEO GAUGES ──────────────────────────────────────────────────────────
    if has_video:
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        st.markdown('<p class="slabel">Body Language</p>', unsafe_allow_html=True)

        g1, g2, g3 = st.columns(3)
        for col, val, title, color, thr in [
            (g1, st.session_state.eye_contact_score, "👁️ Eye Contact",  "#3b82f6", 60),
            (g2, st.session_state.posture_score,      "🧍 Posture",       "#10b981", 60),
            (g3, st.session_state.gesture_score,      "🤚 Gestures",      "#f59e0b", 40),
        ]:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=val,
                title={"text": title, "font": {"size": 14, "color": "#f0f0f0"}},
                number={"font": {"color": "#f0f0f0"}},
                gauge={
                    "axis":  {"range": [0,100], "tickcolor": "#9ca3af"},
                    "bar":   {"color": color},
                    "bgcolor": "#1e1e1e",
                    "steps": [{"range":[0,40], "color":"#3b1010"},
                               {"range":[40,70],"color":"#2a1f0a"},
                               {"range":[70,100],"color":"#0a2a1a"}],
                    "threshold": {"line":{"color":"#ef4444","width":3},
                                  "thickness":0.75,"value":thr},
                }))
            fig.update_layout(
                height=200, margin=dict(l=20,r=20,t=40,b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f0f0f0"}
            )
            col.plotly_chart(fig, use_container_width=True)
            if   val>=70: col.success("✅ Excellent")
            elif val>=50: col.warning("⚠️ Moderate")
            else:         col.error("❌ Needs work")

        if len(st.session_state.eye_contact_history) > 5:
            ec_l = list(st.session_state.eye_contact_history)
            p_l  = list(st.session_state.posture_history)
            tx   = np.linspace(0,
                list(st.session_state.timestamps)[-1] if st.session_state.timestamps else len(ec_l),
                len(ec_l))
            fig  = go.Figure()
            fig.add_trace(go.Scatter(x=tx, y=ec_l, name="Eye Contact",
                line=dict(color="#3b82f6",width=2),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.07)"))
            fig.add_trace(go.Scatter(x=tx, y=p_l, name="Posture",
                line=dict(color="#10b981",width=2)))
            fig.add_hline(y=70, line_dash="dot", line_color="#10b981",
                          annotation_text="Good (70)")
            fig.update_layout(
                xaxis_title="Time (s)", yaxis_title="Score (%)",
                yaxis=dict(range=[0,100]), height=270,
                margin=dict(l=0,r=0,t=20,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,20,20,0.5)",
                font={"color": "#9ca3af"},
                xaxis={"gridcolor": "#2a2a2a"}, yaxis2={"gridcolor": "#2a2a2a"},
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── AUDIO TABS ────────────────────────────────────────────────────────────
    if has_audio:
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        tp, tc, tn, tt_tab = st.tabs(["📈 Pitch","🎯 Confidence","⚠️ Nervousness","💡 Tips"])

        with tp:
            voiced = [(t, p) for t, p in
                      zip(st.session_state.timestamps, st.session_state.pitch_history)
                      if p is not None and np.isfinite(p)]
            if voiced:
                df  = pd.DataFrame(voiced, columns=["time","pitch"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.time, y=df.pitch, mode="lines",
                    line=dict(color="#3b82f6",width=1.5), name="Pitch (Hz)"))
                if len(df) > 10:
                    fig.add_trace(go.Scatter(
                        x=df.time, y=df.pitch.rolling(10,center=True).mean(),
                        mode="lines", line=dict(color="#f59e0b",width=2.5,dash="dash"),
                        name="Trend (10-pt MA)"))
                for m in st.session_state.nervous_moments:
                    fig.add_vline(x=m["time"], line_dash="dash",
                                  line_color="#ef4444",
                                  annotation_text=f"⚠️{m['score']}%",
                                  annotation_position="top")
                fig.update_layout(
                    xaxis_title="Time (s)", yaxis_title="Hz", height=350,
                    margin=dict(l=0,r=0,t=20,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,20,20,0.5)",
                    font={"color": "#9ca3af"},
                    xaxis={"gridcolor": "#2a2a2a"}, yaxis={"gridcolor": "#2a2a2a"},
                )
                st.plotly_chart(fig, use_container_width=True)
                c1,c2,c3 = st.columns(3)
                stab = max(0, 100 - df.pitch.std()/df.pitch.mean()*100) if df.pitch.mean()>0 else 0
                c1.metric("Avg Pitch",   f"{df.pitch.mean():.1f} Hz")
                c2.metric("Pitch Range", f"{df.pitch.max()-df.pitch.min():.1f} Hz")
                c3.metric("Stability",   f"{stab:.1f}%")
            else:
                st.info("No voiced frames. Adjust pitch range or silence threshold.")

        with tc:
            ch = list(st.session_state.confidence_history)
            if ch:
                ts = np.linspace(0,
                    list(st.session_state.timestamps)[-1] if st.session_state.timestamps else len(ch),
                    len(ch))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts, y=ch, mode="lines+markers",
                    fill="tozeroy", fillcolor="rgba(16,185,129,0.09)",
                    line=dict(color="#10b981",width=2.5), marker=dict(size=3),
                    name="Confidence"))
                for thr, col_hex, lbl in [
                    (70,"#10b981","Excellent"),(50,"#f59e0b","Good"),(30,"#ef4444","Fair")]:
                    fig.add_hline(y=thr, line_dash="dot", line_color=col_hex,
                                  annotation_text=f"{lbl} ({thr}%)")
                fig.update_layout(
                    xaxis_title="Time (s)", yaxis_title="Score (%)",
                    yaxis=dict(range=[0,100]), height=350,
                    margin=dict(l=0,r=0,t=20,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,20,20,0.5)",
                    font={"color": "#9ca3af"},
                    xaxis={"gridcolor": "#2a2a2a"}, yaxis2={"gridcolor": "#2a2a2a"},
                )
                st.plotly_chart(fig, use_container_width=True)
                avg_c = float(np.mean(ch))
                trend = ch[-1]-ch[0] if len(ch)>1 else 0
                c1,c2 = st.columns(2)
                c1.metric("Average", f"{avg_c:.0f}%")
                c2.metric("Trend", f"{'↗ +' if trend>=0 else '↘ '}{abs(trend):.0f} pts",
                          delta_color="normal" if trend>=0 else "inverse")
            else:
                st.info("Confidence history will appear here after analysis.")

        with tn:
            nm = st.session_state.nervous_moments
            if nm:
                deduped = []
                for m in nm:
                    if not deduped or m["time"]-deduped[-1]["time"] > 2.0:
                        deduped.append(m)
                st.warning(f"{len(deduped)} nervous episode(s) detected")
                for i, m in enumerate(deduped):
                    with st.expander(f"Episode #{i+1}  ·  {m['time']:.1f}s  ·  score {m['score']}%"):
                        for note in set(m.get("notes",[])):
                            st.write(f"• {note}")
            else:
                st.success("✅ No significant nervousness detected.")

            if nerv>=65:
                st.markdown('<div class="alert">⚠️ <strong>High tension.</strong> '
                    'Breathe in for 4, out for 6 between sentences.</div>',
                    unsafe_allow_html=True)
            elif nerv>=35:
                st.markdown('<div class="alert">ℹ️ <strong>Mild tension.</strong> '
                    'Relax jaw and shoulders.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="good">✅ Calm delivery — '
                    'good emotional control.</div>', unsafe_allow_html=True)

        with tt_tab:
            tips = st.session_state.coaching_tips
            if tips:
                st.write("**Personalised coaching based on your recording:**")
                for tip in tips:
                    st.markdown(f'<div class="tip">{tip}</div>', unsafe_allow_html=True)
            else:
                st.info("Tips will appear here after analysis.")

    # ── OVERALL GRADE — FIX 2: unified compute_overall() ─────────────────────
    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown('<p class="slabel">Overall Grade</p>', unsafe_allow_html=True)

    overall, letter, msg = compute_overall(
        conf=conf, nerv=nerv,
        pitch_history=list(st.session_state.pitch_history),
        timestamps=list(st.session_state.timestamps),
        eye=st.session_state.eye_contact_score,
        pos=st.session_state.posture_score,
        gest=st.session_state.gesture_score,
        has_video=has_video,
        video_weight_pct=vid_wt,
    )

    # Compute audio/video sub-scores for display
    vp   = [p for p in st.session_state.pitch_history if p is not None and np.isfinite(p)]
    tt   = max(list(st.session_state.timestamps), default=1)
    sp_t = len(vp) * SpeechAnalyzer.HOP
    a_sc = float(np.clip(conf - nerv*0.4 + min(sp_t/max(tt,1)*20, 10), 0, 100))
    v_sc = (st.session_state.eye_contact_score*0.40
            + st.session_state.posture_score*0.40
            + st.session_state.gesture_score*0.20) if has_video else 0.0

    _, mid, _ = st.columns([1,2,1])
    with mid:
        vline = (f'<div style="font-size:.8rem;color:#9ca3af;margin:.4rem 0">'
                 f'🎤 Voice {int(a_sc)} &nbsp;·&nbsp; 📹 Video {int(v_sc)}</div>'
                 ) if has_video else ""
        st.markdown(
            f'<div class="grade-wrap">'
            f'<div class="grade-letter">{letter}</div>'
            f'<div class="grade-score">{overall:.1f} / 100</div>'
            f'{vline}'
            f'<div class="grade-msg">{msg}</div>'
            f'</div>', unsafe_allow_html=True)

    # ── EXPORT — FIX 2: JSON also uses compute_overall() ─────────────────────
    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown('<p class="slabel">Export</p>', unsafe_allow_html=True)

    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        report = {
            "generated": datetime.now().isoformat(),
            "audio": {
                "confidence":          conf,
                "nervousness":         nerv,
                "nervous_episodes":    len(st.session_state.nervous_moments),
                "filler_count":        st.session_state.filler_count,
                "filler_rate_per_min": st.session_state.filler_rate,
            },
            "video": {
                "eye_contact": st.session_state.eye_contact_score,
                "posture":     st.session_state.posture_score,
                "gesture":     st.session_state.gesture_score,
            } if has_video else None,
            "overall":        overall,          # ← FIX 2: same formula as screen
            "grade":          letter,
            "grade_message":  msg,
            "coaching_tips":  st.session_state.coaching_tips,
        }
        st.download_button("📄 JSON Report",
            json.dumps(report, indent=2),
            f"speaksmart_{datetime.now():%Y%m%d_%H%M%S}.json",
            "application/json", use_container_width=True)

    with ex2:
        if st.button("📑 Generate PDF", use_container_width=True):
            with st.spinner("Building PDF…"):
                pdf_bytes = generate_pdf_report()
            st.download_button("⬇️ Download PDF", pdf_bytes,
                f"speaksmart_{datetime.now():%Y%m%d_%H%M%S}.pdf",
                "application/pdf", use_container_width=True)

    with ex3:
        if st.session_state.pitch_history:
            ph = list(st.session_state.pitch_history)
            df_csv = pd.DataFrame({
                "time_s":   list(st.session_state.timestamps)[:len(ph)],
                "pitch_hz": [p if (p is not None and np.isfinite(p)) else np.nan for p in ph],
                "rms":      list(st.session_state.rms_history)[:len(ph)],
            })
            st.download_button("📊 Pitch CSV",
                df_csv.to_csv(index=False),
                f"speaksmart_pitch_{datetime.now():%Y%m%d_%H%M%S}.csv",
                "text/csv", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="div">', unsafe_allow_html=True)
st.markdown('<p class="slabel">Exercises & Resources</p>', unsafe_allow_html=True)

r1,r2,r3,r4 = st.tabs(["🎵 Voice","📹 Video Tips","📚 Reading","❓ FAQ"])

with r1:
    for cat, items in {
        "🫁 Breathing":     ["4-7-8 (inhale 4, hold 7, exhale 8)",
                              "Diaphragmatic breathing (belly, not chest)",
                              "Box breathing (4 counts each side)"],
        "🎶 Warm-ups":      ["Lip trills for 10 s","Humming scales","Jaw stretches"],
        "🗣️ Articulation": ["Tongue twisters at 3 speed levels",
                              "Exaggerated vowels while reading aloud",
                              "Minimal-pair drills (ship/sheep, bit/beat)"],
    }.items():
        with st.expander(cat):
            for it in items: st.write(f"• {it}")

with r2:
    for cat, items in {
        "👁️ Eye Contact": ["Look at the camera LENS, not your face preview",
            "Put a coloured dot above the lens as a focus point",
            "Aim for 70-80% camera time; glances away are natural"],
        "🧍 Posture":      ["Ears over shoulders over hips",
            "Use a riser to get your camera to eye level",
            "Record a side-profile once to check forward-head posture"],
        "🤚 Gestures":     ["Open palms = confidence; clasped = tension",
            "Keep gestures within shoulder width",
            "Avoid touching face/neck — these are nervousness cues"],
        "📷 Camera Setup": ["Camera at exact eye level",
            "Soft front lighting (ring light or window in front)",
            "Plain or blurred background"],
    }.items():
        with st.expander(cat):
            for it in items: st.write(f"• {it}")

with r3:
    with st.expander("📖 Books"):
        for b in ['"Talk Like TED" — Carmine Gallo',
                  '"The Charisma Myth" — Olivia Fox Cabane',
                  '"Presence" — Amy Cuddy',
                  '"Never Split the Difference" — Chris Voss']:
            st.write(f"• {b}")
    with st.expander("🏢 Practice"):
        for o in ["Toastmasters International (free guest visits)",
                  "Record and review yourself weekly",
                  "Volunteer for presentations at work or class"]:
            st.write(f"• {o}")

with r4:
    for q,a in {
        "How is pitch extracted?":
            "LibROSA YIN on 250 ms windows. Median across frames removes octave errors.",
        "What does confidence measure?":
            "4 sub-scores: pitch stability (CV), volume consistency, speaking continuity, and pitch comfort zone.",
        "What does nervousness measure?":
            "Pitch CV + jitter + baseline elevation + RMS inconsistency + silence ratio — all weighted by sensitivity.",
        "How does eye contact work?":
            "Iris offset from each eye's midpoint, divided by inter-ocular distance (IOD), so head turns don't crash the score.",
        "How does posture work?":
            "All thresholds are fractions of shoulder width — camera-distance independent.",
        "Why does gesture score show 50?":
            "50 = neutral (hands out of frame). 30 = wrists visible but stiff. >50 = open expressive hands.",
        "How do filler words get detected?":
            "With openai-whisper installed, we transcribe then regex-match 20+ common filler patterns.",
        "Is my data stored?":
            "No. Everything runs locally. Nothing leaves your machine.",
        "Why do PDF, JSON, and screen now show the same score?":
            "v3.0-fixed extracts compute_overall() as a single shared function. "
            "All three exports call it with identical arguments, so the grade is always consistent.",
        "Required packages?":
            "Core: librosa, streamlit, plotly, numpy, pandas, reportlab. "
            "Video: mediapipe, opencv-python-headless. "
            "Fillers: openai-whisper. Live: streamlit-webrtc, aiortc. Mic: sounddevice, soundfile.",
    }.items():
        with st.expander(q): st.write(a)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="div">', unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center;color:#4b5563;font-size:.78rem;line-height:1.9">
  🗣️ <strong style="color:#9ca3af">SpeakSmart AI v3.0-fixed</strong> — Your Personal Communication Coach<br>
  Project Exhibition-I · Group 256<br>
  Raunak Kumar Modi · Jahnvi Pandey · Rishi Singh Shandilya · Unnati Lohana · Vedant Singh
</p>""", unsafe_allow_html=True)