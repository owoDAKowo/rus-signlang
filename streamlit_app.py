import streamlit as st
from streamlit_webrtc import webrtc_streamer
from Recognizer import SignLanguageRecognizer

if __name__ == "__main__":
    st.title('Распознавание жестового языка')

    webrtc_streamer(
        key="sign-language-recognizer",
        rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.cloudflare.com:3478",
                                 "stun:stun.l.google.com:19302"]}],
        },
        video_processor_factory = SignLanguageRecognizer,
        media_stream_constraints = {"video": True, "audio": False},
    )