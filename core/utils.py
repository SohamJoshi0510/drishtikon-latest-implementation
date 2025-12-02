import os
import sys

# ================================================================
#  BASE DIRECTORY (ROOT OF PROJECT)
#  This ensures file paths remain stable regardless of how script
#  is launched (Tkinter, VSCode, Command Prompt, etc.)
# ================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def absolute_path(*paths) -> str:
    """
    Returns a safe absolute path from BASE_DIR.
    Example: absolute_path("reading", "cred", "key.json")
    """
    return os.path.join(BASE_DIR, *paths)


# ================================================================
#  CREDENTIAL LOADING HELPER
# ================================================================
def load_credential_path(module_folder: str, filename: str) -> str:
    """
    Returns the absolute path to a credential JSON located inside a module folder.

    Example:
        load_credential_path("reading", "reading-key.json")
        → <BASE_DIR>/reading/cred/reading-key.json
    """
    return absolute_path(module_folder, "cred", filename)


# ================================================================
#  DIRECTORY ENSURER
# ================================================================
def ensure_dir(path: str):
    """
    Creates a directory if it does not exist.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"[UTIL] Failed to create directory {path}: {e}", file=sys.stderr)


# ================================================================
#  GENERAL HELPER: PRINT FILE PATH FOR DEBUGGING
# ================================================================
def debug_path(label: str, path: str):
    print(f"[DEBUG PATH] {label}: {path}")

if __name__ == "__main__":
    print(f"[UTIL] BASE_DIR set to: {BASE_DIR}")