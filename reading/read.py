import os
import sys

# Ensure the project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import time
import datetime
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import io
from dotenv import load_dotenv
import google.generativeai as genai

from core.utils import absolute_path, ensure_dir, load_credential_path
from core.tts import speak
from core.logger import log

load_dotenv()

# ================================================================
#  CREDENTIALS (local to reading/)
# ================================================================
CRED_PATH = load_credential_path("reading", "reading-key.json")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ================================================================
#  HELPERS
# ================================================================
def ensure_results_dir():
    ensure_dir(absolute_path("results"))
    ensure_dir(absolute_path("results", "reading_outputs"))


# ================================================================
#  IMAGE OPTIMIZATION (very important for latency)
# ================================================================
def optimize_image(image_path):
    """
    Resizes image to ~1800px max dimension and compresses to JPEG.
    Handles PNGs or any RGBA image by converting to RGB.
    """
    img = Image.open(image_path)

    # Convert images with alpha channel (RGBA) to RGB
    if img.mode == "RGBA":
        img = img.convert("RGB")

    img.thumbnail((1800, 1800))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)

    return buf.getvalue()


# ================================================================
#  GEMINI OCR + SUMMARIZER
# ================================================================
def gemini_read(image_path, prompt):
    """
    Runs Gemini OCR + summarization on an image.
    Uses streaming mode for partial output.
    """

    if not GEMINI_API_KEY or not GEMINI_MODEL:
        return "Gemini not configured.", 0

    optimized_bytes = optimize_image(image_path)

    model = genai.GenerativeModel(GEMINI_MODEL)

    mime_type = "image/jpeg"
    final_text = []

    t0 = time.time()

    response = model.generate_content(
        [
            {"mime_type": mime_type, "data": optimized_bytes},
            prompt
        ],
        stream=True
    )

    for chunk in response:
        if hasattr(chunk, "text") and chunk.text:
            final_text.append(chunk.text)

    t1 = time.time()
    duration = round(t1 - t0, 2)

    text_output = "".join(final_text)
    return text_output, duration


# ================================================================
#  FILE SELECTION
# ================================================================
def choose_file():
    root = tk.Tk()
    # root.withdraw()

    fp = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All Files", "*.*")
        ]
    )

    img = cv2.imread(fp)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = absolute_path("results", "reading_outputs", f"capture_{ts}.jpg")
    cv2.imwrite(path, img)

    root.destroy()
    return fp


# ================================================================
#  CAMERA FALLBACK (if user cancels file picker)
# ================================================================
def capture_image():
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        speak("Camera not found.")
        return None

    speak("Press SPACE to capture, or ESC to exit.")

    img = None
    path = None

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        cv2.imshow("Camera Capture - Press SPACE", frame)
        key = cv2.waitKey(1)

        if key == 32:  # Spacebar
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = absolute_path("results", "reading_outputs", f"capture_{ts}.jpg")
            cv2.imwrite(path, frame)
            img = frame
            break

        elif key == 27:  # ESC
            break

    cam.release()
    cv2.destroyAllWindows()
    return path


# ================================================================
#  MAIN EXECUTION FUNCTION
# ================================================================
def main():
    ensure_results_dir()

    speak("Select an image file. If you cancel, I will open the camera.")

    # 1. File chooser
    img_path = choose_file()

    # 2. If user cancels, go to camera mode
    if not img_path:
        speak("No file selected. Switching to camera.")
        img_path = capture_image()

    if not img_path:
        speak("No image captured. Exiting.")
        return

    # 3. Gemini prompt (customizable)
    refinement_prompt = f"""
                Following is an image captured by a blind person smartphone.. please help me read this book by extracting the exact text content. Do not change a word.
                    """

    speak("Processing the image. Please wait.")

    text, duration = gemini_read(img_path, refinement_prompt)

    log("READING", img_path, f"{len(text)} chars", duration)

    print("\n===== OCR RESULT =====\n")
    print(text)
    print("\n=======================\n")

    speak(text)


# ================================================================
if __name__ == "__main__":
    main()
