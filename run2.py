import streamlit as st
import cv2
import pyttsx3
import time
from ultralytics import YOLO
from translate import Translator
from PIL import Image, ImageOps
import speech_recognition as sr
import os
import warnings
import threading
import queue

warnings.filterwarnings("ignore")

# Common variables
image_dir = "images/"
target_size = (150, 150)

# Sidebar menu
app_mode = st.sidebar.selectbox(
    "Choose a feature",
    ["Sign Language Detection", "Sign Language Translator"]
)

# Shared utility: Text-to-Speech worker function
def tts_worker(word_queue, translator):
    while True:
        word, lang_choice = word_queue.get()
        if word is None:  # Exit signal
            break

        translated_word = word
        if lang_choice == 'French' and translator:
            translated_word = translator.translate(word)

        # Create and configure TTS engine
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.setProperty('volume', 1.0)

        print(f"Speaking: {translated_word}")
        tts_engine.say(translated_word)
        tts_engine.runAndWait()
        tts_engine.stop()
        word_queue.task_done()

# Sign Language Detection Feature
if app_mode == "Sign Language Detection":
    st.title("Sign Language Detection")

    # YOLO model setup
    model = YOLO('best.pt')
    label_map = {i: chr(65 + i) for i in range(26)}

    # Language selection
    language_choice = st.sidebar.radio("Select your language:", ["English"])
    translator = Translator(to_lang="en") if language_choice == 'English' else None

    # Queue for TTS
    word_queue = queue.Queue()

    # Start TTS thread
    tts_thread = threading.Thread(target=tts_worker, args=(word_queue, translator), daemon=True)
    tts_thread.start()

    # Detection settings
    start_detection = st.sidebar.button("Start Detection")
    if start_detection:
        st.markdown("## Detection in Progress... Press **Stop** to end.")

        word = ""
        current_letter = None
        hold_time_threshold = 0.5
        idle_time_threshold = 5
        last_detection_time = time.time()
        current_letter_time = 0

        cap = cv2.VideoCapture(0)
        video_placeholder = st.empty()
        stop_detection = st.button("Stop Detection")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break

            results = model(frame)
            detected_letter = None
            for result in results:
                if len(result.boxes) > 0:
                    box = result.boxes[0]
                    detected_label = int(box.cls[0].item())
                    detected_letter = label_map.get(detected_label)
                    last_detection_time = time.time()
                    break

            if detected_letter == current_letter:
                current_letter_time += 1 / 30
            else:
                current_letter = detected_letter
                current_letter_time = 0

            if current_letter_time >= hold_time_threshold and current_letter is not None:
                word += current_letter
                current_letter = None
                current_letter_time = 0
                print(f"Current Word: {word}")

            if time.time() - last_detection_time >= idle_time_threshold and word:
                print(f"Queueing Word for TTS: {word}")
                word_queue.put((word, language_choice))
                word = ""

            cv2.putText(frame, f"Current Letter: {current_letter if current_letter else 'None'}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"Word: {word}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            video_placeholder.image(pil_image, caption="Sign Language Detection", use_container_width=True)

            if stop_detection:
                break

        cap.release()
        word_queue.put((None, None))  # Signal TTS thread to exit
        word_queue.join()
        st.markdown("## Detection Stopped.")

# Simple Sign Language Image Viewer Feature
elif app_mode == "Sign Language Translator":
    st.title("Sign Language Translator")

    r = sr.Recognizer()
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ''

    # Speech-to-text feature
    if st.button("Speak"):
        with sr.Microphone() as source:
            st.write("Speak now...")
            audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            st.session_state.text_input = text
            st.write(f"You said: {text}")
        except sr.UnknownValueError:
            st.write("Could not understand audio")
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Show images corresponding to spoken or typed input
    def show_images(text):
        images = [f"{image_dir}{char}.jpg" for char in text.lower() if os.path.exists(f"{image_dir}{char}.jpg")]
        if not images:
            st.write("No images found for that input.")
            return

        if 'current_image_index' not in st.session_state:
            st.session_state.current_image_index = 0

        num_images = len(images)

        if num_images > 0:
            col1, col2, col3 = st.columns((1, 4, 1))

            if col1.button("<--"):
                st.session_state.current_image_index = (st.session_state.current_image_index - 1) % num_images

            try:
                img = Image.open(images[st.session_state.current_image_index])
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                border = (10, 10, 10, 10)
                img = ImageOps.expand(img, border=border, fill="white")
                col2.image(img, use_column_width=True, caption=f"Image {st.session_state.current_image_index + 1}/{num_images}")
            except (IndexError, FileNotFoundError, OSError) as e:
                col2.error(f"Error displaying image: {e}")

            if col3.button("-->"):
                st.session_state.current_image_index = (st.session_state.current_image_index + 1) % num_images

    # Display images based on user input
    show_images(st.session_state.text_input)
