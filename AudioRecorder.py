import sys
import pyaudio
from pynput import keyboard
import wave


class AudioRecorder(keyboard.Listener):
    def __init__(self, rate, chunk, wav_output_filename):
        super(AudioRecorder, self).__init__(self.on_press, self.on_release)
        self.recorder = Recorder(rate, chunk, wav_output_filename)

    def on_press(self, key):
        try:
            if key.char == 'r':
                self.recorder.start()
            return True
        except AttributeError:
            if key == keyboard.Key.esc:
                sys.exit(0)

    def on_release(self, key):
        try:
            if key.char == 'r':
                self.recorder.stop()
                return False
        except AttributeError:
            if key == keyboard.Key.esc:
                sys.exit(0)


class Recorder:
    def __init__(self, rate, chunk, wav_output_filename):
        self._rate = rate
        self._chunk = chunk
        self.channels = 1
        self.wav_output_filename = wav_output_filename
        self.format = pyaudio.paInt16
        self._audio_interface = pyaudio.PyAudio()
        self.recording = False
        self.audio_stream = None
        self.frames = []

    def callback(self, in_data, frame_count, time_info, status):
        self.frames.append(in_data)
        return in_data, pyaudio.paContinue

    def start(self):
        if self.recording: return
        # Reset buffer
        self.frames = []
        self.audio_stream = self._audio_interface.open(
            format=self.format,
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self.callback,
        )
        print("\nStarted recording")
        self.recording = True

    def stop(self):
        if not self.recording: return
        self.recording = False
        print("Stopped recording")
        self.audio_stream.stop_stream()
        self.audio_stream.close()

        # Using with will close the file automatically
        with wave.open(self.wav_output_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self._audio_interface.get_sample_size(self.format))
            wf.setframerate(self._rate)
            wf.writeframes(b''.join(self.frames))
