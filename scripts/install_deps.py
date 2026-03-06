import subprocess
import sys

def install():
    libraries = [
        "ultralytics", "PyQt5", "opencv-python", 
        "speech_recognition", "PyAudio", "torch", "torchvision"
    ]
    
    print("🚀 Starting installation of dependencies...")
    for lib in libraries:
        print(f"Installing {lib}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
    print("✅ All done!")

if __name__ == "__main__":
    install()