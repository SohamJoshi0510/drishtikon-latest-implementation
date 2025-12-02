import sys
import os

# Ensure the project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import subprocess
from core.stt import listen
from core.tts import speak
from core.logger import log
from core.utils import absolute_path

# ================================================================
#   MAIN CONTROLLER
#   Controls the entire system flow:
#   - STT listens for commands
#   - "read"   → reading/read.py
#   - "detect" → yolo/detect.py
#   - "exit"   → quits
# ================================================================

def run_module(relative_path):
    """
    Runs a Python module (like reading/read.py) as a subprocess.
    Always uses an absolute path.
    """

    target = absolute_path(relative_path)

    if not os.path.exists(target):
        speak(f"Module {relative_path} not found.")
        log("MAIN", relative_path, "Module missing")
        return

    try:
        subprocess.run([sys.executable, target])
    except Exception as e:
        speak("Failed to launch module.")
        log("MAIN", relative_path, f"Launch error: {e}")


def main():
    speak("System ready. Say a command: read, detect, or exit.")
    print("\n[MAIN] Awaiting voice commands...")

    while True:
        cmd = listen()

        if not cmd:
            continue

        cmd = cmd.lower().strip()
        print(f"[MAIN] Heard command: {cmd}")

        # ===========================
        #  MODULE: READING
        # ===========================
        if "read" in cmd or "reading" in cmd:
            speak("Opening reading module.")
            log("MAIN", "-", "Launching reading module")
            run_module("reading/read.py")
            continue

        # ===========================
        #  MODULE: OBJECT DETECTION
        # ===========================
        if "detect" in cmd or "object" in cmd:
            speak("Opening object detection module.")
            log("MAIN", "-", "Launching YOLO module")
            run_module("yolo/detect.py")
            continue

        # ===========================
        #  EXIT SYSTEM
        # ===========================
        if "exit" in cmd or "quit" in cmd:
            speak("Goodbye.")
            log("MAIN", "-", "System exit")
            break

        # ===========================
        #  UNKNOWN COMMAND
        # ===========================
        speak("I didn't understand that.")
        log("MAIN", "-", f"Unknown command: {cmd}")


# ================================================================
if __name__ == "__main__":
    main()
