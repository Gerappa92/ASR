import sounddevice as sd
import soundfile as sf

# The sample rate defines how many samples of the audio are taken per second
sample_rate = 16000  
duration = 10  # The duration of the recording in seconds

def record_audio(duration, sample_rate):
    print("Recording...")
    my_recording = sd.rec(int(duration * sample_rate), 
                         samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished")
    return my_recording

if __name__ == "__main__":
    my_recording = record_audio(duration, sample_rate)
    # This will save the recording in a .wav file
    sf.write('my_recording.wav', my_recording, sample_rate)