import sys
import os
import subprocess
import threading
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.stt import listen
from core.tts import speak
from core.logger import log
from core.utils import absolute_path


# ================================================================
#   HOTKEY STOP SYSTEM (Windows Only)
# ================================================================

active_processes = []


def emergency_stop():
    """Kills all modules instantly and exits."""
    print("[STOP] Emergency STOP triggered (ESC pressed)!")
    speak("Stopping all modules.")
    kill_all_processes()
    os._exit(0)


def windows_hotkey_listener():
    """Listen for ESC hotkey and trigger STOP."""
    import keyboard

    keyboard.add_hotkey("esc", emergency_stop)
    print("[STOP] ESC hotkey listener active (Windows).")

    # keep thread alive
    while True:
        time.sleep(1)


def kill_all_processes():
    """Force-kills all subprocesses."""
    print("[STOP] Killing all subprocesses...")

    for p in active_processes[:]:
        try:
            p.terminate()
            time.sleep(0.2)
            p.kill()
        except:
            pass

    active_processes.clear()

    # Kill OpenCV windows
    try:
        import cv2
        cv2.destroyAllWindows()
    except:
        pass

    print("[STOP] All subprocesses terminated.")


# ================================================================
#   MODULE LAUNCHER
# ================================================================

def start_process(relative_path):
    """Launch reading/yolo modules as subprocesses and track them."""
    target = absolute_path(relative_path)

    if not os.path.exists(target):
        speak(f"Module {relative_path} not found.")
        log("MAIN", relative_path, "Missing module")
        return

    try:
        p = subprocess.Popen([sys.executable, target])
        active_processes.append(p)

        # Wait until process closes
        while p.poll() is None:
            time.sleep(0.1)

        active_processes.remove(p)

    except Exception as e:
        log("MAIN", relative_path, f"Launch error: {e}")
        speak("Could not launch module.")


# ================================================================
#   MAIN LOOP
# ================================================================

def main():
    speak("System ready. Say read, detect, or exit.")
    print("[MAIN] System awaiting commands...")

    # Start STOP listener thread (ESC key)
    threading.Thread(target=windows_hotkey_listener, daemon=True).start()

    while True:
        cmd = listen()
        if not cmd:
            continue

        cmd = cmd.lower().strip()
        print(f"[MAIN] Heard: {cmd}")

        # Reading module
        if "read" in cmd:
            speak("Opening reading module.")
            log("MAIN", "-", "Launch reading")
            start_process("reading/read.py")
            continue

        # YOLO module
        if "detect" in cmd or "object" in cmd:
            speak("Opening object detection module.")
            log("MAIN", "-", "Launch YOLO")
            start_process("yolo/detect.py")
            continue

        # Exit system
        if "exit" in cmd or "quit" in cmd:
            speak("Goodbye.")
            kill_all_processes()
            break

        # Unknown command
        speak("I did not understand.")
        log("MAIN", "-", f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
