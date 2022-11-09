import sys
import threading
import pyaudio
from pynput import keyboard
import wave
import cv2 as cv


class Recorder(keyboard.Listener):
    def __init__(self, rate, chunk, wav_output_filename, avi_output_filename):
        super(Recorder, self).__init__(self.on_press, self.on_release)
        self.pressed = False
        self.video_recorder = VideoRecoder(avi_output_filename)
        self.audio_recorder = AudioRecorder(rate, chunk, wav_output_filename)

    def on_press(self, key):
        try:
            if key.char == 'r' and not self.pressed:
                self.pressed = True
                self.video_recorder.start()
                self.audio_recorder.start()
                print("\nStarted recording")
            return True
        except AttributeError:
            if key == keyboard.Key.esc:
                sys.exit(0)

    def on_release(self, key):
        try:
            if key.char == 'r':
                self.video_recorder.stop()
                self.audio_recorder.stop()
                print("Stopped recording")
                return False
        except AttributeError:
            if key == keyboard.Key.esc:
                sys.exit(0)

    def start(self):
        self.video_recorder.start()
        self.audio_recorder.start()
        print("\nStarted recording")

    def stop(self):
        self.video_recorder.stop()
        self.audio_recorder.stop()
        print("Stopped recording")


class VideoRecoder:
    def __init__(self, avi_output_filename):
        self.duration = 0
        self.open = True
        self.fps = 30  # fps should be the minimum constant rate at which the camera can
        self.fourcc = "DIVX"  # capture images (with no decrease in speed over time; testing is required)
        self.frameSize = (640, 480)  # video formats and sizes also depend and vary according to the camera used
        self.video_filename = avi_output_filename
        self.video_cap = cv.VideoCapture(0)
        self.video_writer = cv.VideoWriter_fourcc(*self.fourcc)
        self.video_out = cv.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
        self.frame_counts = 1

    def record(self):
        while self.open:
            ret, video_frame = self.video_cap.read()
            if ret:
                self.video_out.write(video_frame)
                self.frame_counts += 1
            else:
                break
        self.video_cap.release()
        self.video_out.release()
        cv.destroyAllWindows()

    def start(self):
        video_thread = threading.Thread(target=self.record)
        video_thread.start()

    def stop(self):
        self.open = False


class AudioRecorder:
    def __init__(self, rate, chunk, wav_output_filename):
        self.duration = 0
        self.open = True
        self.rate = rate
        self.chunk = chunk
        self.channels = 1
        self.wav_output_filename = wav_output_filename
        self.format = pyaudio.paInt16
        self.audio_interface = pyaudio.PyAudio()
        self.frames = []
        self.audio_stream = self.audio_interface.open(
            format=self.format,
            channels=1, rate=self.rate,
            input=True, frames_per_buffer=self.chunk,
        )

    def record(self):
        self.audio_stream.start_stream()
        while self.open:
            try:
                self.duration += self.chunk / self.rate
                data = self.audio_stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                print(e)

        self.audio_stream.stop_stream()
        self.audio_stream.close()

        # Using with will close the file automatically
        with wave.open(self.wav_output_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio_interface.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))

    def start(self):
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()

    def stop(self):
        self.open = False
