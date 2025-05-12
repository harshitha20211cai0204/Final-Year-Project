import streamlit as st
import speech_recognition as sr
import os
from PIL import Image, ImageOps
import warnings

warnings.filterwarnings("ignore")

image_dir = "images/"
target_size = (150, 150)

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

        # Add arrow buttons for navigation
        arrow_left = "<--"
        arrow_right = "-->"

        # Previous image arrow
        if col1.button(arrow_left):
            st.session_state.current_image_index = (st.session_state.current_image_index - 1) % num_images

        # Display the current image
        try:
            img = Image.open(images[st.session_state.current_image_index])
            img = img.resize(target_size, Image.Resampling.LANCZOS)  # Correct resampling filter
            border = (10, 10, 10, 10)
            img = ImageOps.expand(img, border=border, fill="white")

            col2.image(img, use_column_width=True, caption=f"Image {st.session_state.current_image_index + 1}/{num_images}")
        except (IndexError, FileNotFoundError, OSError) as e:
            col2.error(f"Error displaying image: {e}")

        # Next image arrow
        if col3.button(arrow_right):
            st.session_state.current_image_index = (st.session_state.current_image_index + 1) % num_images

st.title("Simple Sign Language Image Viewer")

# Speech recognition setup
r = sr.Recognizer()
if 'text_input' not in st.session_state:
    st.session_state.text_input = ''


# Microphone button
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

# Image display        
show_images(st.session_state.text_input)
