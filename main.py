import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer, VitsModel, AutoTokenizer 
from transformers.utils import logging
import torch

import numpy as np
import sounddevice as sd
import threading
import keyboard

import time

logging.set_verbosity_error()
torch.set_default_device("cuda")

DEBUG = True
DEBUG_FILE_NAME = "log.txt"
DEBUG_FILE = None
def debug_print(string):
    if DEBUG:
        print(string)
        DEBUG_FILE.write(f"{string}\n")

# taken directly from blender according to https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
# I'm gonna switch some of these around later
class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Teffie:
    # 16000 is the sample rate for whisper and must be followed by low-level code
    sd.default.samplerate = 16000
    _FRAMES_PER_SECOND = 16000
    # fiddle with this to change chunk size
    _SECONDS_PER_CHUNK = 0.1
    FRAMES_PER_CHUNK = int(_FRAMES_PER_SECOND * _SECONDS_PER_CHUNK)
    # forcing 1 channel minimizes data size, which will be helpful for speed
    sd.default.channels = 1
    # sd.default.channels is a tuple, so just grab the first and second each
    NUM_INPUT_CHANNELS = sd.default.channels[0]
    NUM_OUTPUT_CHANNELS = sd.default.channels[1]
    NUMPY_DTYPE = np.float32
    NUM_HANDLES = 1024

    def __init__(self):
        debug_print(f"{colors.WARNING}FRAMES_PER_CHUNK is {self.FRAMES_PER_CHUNK}{colors.ENDC}")
        debug_print(f"{colors.WARNING}NUM_INPUT_CHANNELS is {self.NUM_INPUT_CHANNELS}{colors.ENDC}")
        debug_print(f"{colors.WARNING}NUM_OUTPUT_CHANNELS is {self.NUM_OUTPUT_CHANNELS}{colors.ENDC}")
        debug_print(f"{colors.WARNING}NUMPY_DTYPE is {self.NUMPY_DTYPE}{colors.ENDC}")
        debug_print(f"{colors.WARNING}NUM_HANDLES is {self.NUM_HANDLES}{colors.ENDC}")

        print(f"{colors.OKCYAN}Loading audio transcription model...{colors.ENDC}")
        self._speech_transcriber = whisper.load_model("base.en")
        print(f"{colors.OKCYAN}Done loading audio transcription model{colors.ENDC}")
        
        # NOTE: these load from the internet, which is not ideal with our school's crappy wifi
        # preferably to load from cache
        print(f"{colors.OKCYAN}Loading text generation model...{colors.ENDC}")
        self._text_generator = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype="auto")
        print(f"{colors.OKCYAN}Done loading text generation model{colors.ENDC}")
        self._text_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

        print(f"{colors.OKCYAN}Loading speech generation model...{colors.ENDC}")
        self._speech_generator = VitsModel.from_pretrained("facebook/mms-tts-eng") 
        self._speech_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        print(f"{colors.OKCYAN}Done loading speech generation model{colors.ENDC}")

        self._should_stop_recording = False

    # NOTE: this is very hacky and should be swapped out when we get a chance
    # at the very least use a coroutine man!
    # NOTE: is confusing if we go 30 seconds because then we need to hit enter to stop this thread
    # before hitting enter again to start recording.
    # We just need to rearchitect this listener.
    def _keyboard_listener(self):
        time.sleep(0.5)
        input()
        debug_print(f"{colors.WARNING}Recording stopped by keyboard interrupt{colors.ENDC}")
        self._should_stop_recording = True
    
    def record_audio(self):
        input(f"{colors.HEADER}Press ENTER to start recording.{colors.ENDC}")
        print(f"{colors.OKBLUE}Starting recording...{colors.ENDC}")
        time.sleep(0.2)
        print(f"{colors.HEADER}Press ENTER to stop recording. Recording will automatically end at 30 seconds.{colors.ENDC}")
        input_stream = sd.InputStream(callback=None, channels=self.NUM_INPUT_CHANNELS, latency='low', dtype=self.NUMPY_DTYPE)
        
        buffer = np.empty((0, self.NUM_INPUT_CHANNELS))
        self._should_stop_recording = False
        threading.Thread(target=self._keyboard_listener).start()
        input_stream.start()
        while not self._should_stop_recording:
            chunk, overflowed = input_stream.read(self.FRAMES_PER_CHUNK)
            buffer = np.append(buffer, chunk, axis=0) 
            if np.size(buffer, 0) > self._FRAMES_PER_SECOND * 30:
                self._should_stop_recording = True
        input_stream.stop()
        # print(buffer)
        print(f"{colors.OKBLUE}Finished recording{colors.ENDC}")

        return buffer

    def transcribe_audio(self, next_input):
        # applies all transformations to np array necessary to pass into whisper model 
        # referencing https://github.com/openai/whisper/discussions/450
        def transform_buffer(buffer):
            # x = torch.from_numpy(buffer.flatten() / 32768.0).float()
            # print(x.shape)
            # print(whisper.pad_or_trim(x).shape)
            return whisper.pad_or_trim(torch.from_numpy(buffer.flatten()).float())
        decoding_options = whisper.DecodingOptions(language="en")
        print(f"{colors.OKBLUE}Starting transcription...{colors.ENDC}")
        start = time.time()
        next_input = transform_buffer(next_input)
        mel = whisper.log_mel_spectrogram(next_input).to(self._speech_transcriber.device)
        transcription = whisper.decode(self._speech_transcriber, mel, decoding_options).text
        print(f"{colors.OKGREEN}Transcription is \"{transcription}\"{colors.ENDC}")
        end = time.time()
        debug_print(f"{colors.WARNING}Time elapsed during response generation: {end - start}{colors.ENDC}")
        print(f"{colors.OKBLUE}Finished transcription{colors.ENDC}")
        return transcription

    def _truncate_response(self, response):
        # step 1: truncate up to "Answer:"
        answer_start_index = response.index("Answer: ")
        response = response[answer_start_index + len("Answer: "):]

        # step 2: convert last stopping punctuation mark (period or comma) to period and truncate after 
        last_puncutation_mark_index = max(response.rfind(mark) for mark in ['.', ','])
        # AFAIK Python doesn't let you change one character of a string, so this is a workaround 
        response = response[:last_puncutation_mark_index] + '.'
        return response

    def generate_response(self, transcription):
        print(f"{colors.OKBLUE}Starting response generation...{colors.ENDC}")
        start = time.time()
        inputs = self._text_tokenizer(f'''In one paragraph, describe a scene where {transcription} Answer:''', return_tensors="pt", return_attention_mask=False)

        outputs = self._text_generator.generate(**inputs, max_new_tokens=64, do_sample=False)

        response = self._text_tokenizer.batch_decode(outputs)[0]
        response = self._truncate_response(response)
        print(f"{colors.OKGREEN}Generated response: \"{response}\"{colors.ENDC}")
        end = time.time()
        debug_print(f"{colors.WARNING}Time elapsed during response generation: {end - start}{colors.ENDC}")
        print(f"{colors.OKBLUE}Finished response generation{colors.ENDC}")
        return response

    def generate_and_play_audio(self, response):
        inputs = self._speech_tokenizer(response, return_tensors="pt")

        with torch.no_grad():
            output = self._speech_generator(**inputs).waveform

        current_frame = 0

        # sd.default.samplerate = speech_model.config.sampling_rate
        sd.default.channels = 1

        data = np.swapaxes(output.cpu().float().numpy(), 0, 1) 
        fs = self._speech_generator.config.sampling_rate
        event = threading.Event()

        def callback(outdata, frames, time, status):
            nonlocal current_frame
            if status:
                print(status)
            chunksize = min(len(data) - current_frame, frames)
            outdata[:chunksize] = data[current_frame:current_frame + chunksize]
            if chunksize < frames:
                outdata[chunksize:] = 0
                raise sd.CallbackStop()
            current_frame += chunksize

        stream = sd.OutputStream(
            samplerate=fs, callback=callback, finished_callback=event.set)
        with stream:
            event.wait()

    def run(self):
        while True:
            recorded_audio = self.record_audio()
            transcription = self.transcribe_audio(recorded_audio)
            response = self.generate_response(transcription)
            self.generate_and_play_audio(response)


if __name__ == "__main__":
    try:
        print(f"{colors.HEADER}Running Teffie...{colors.ENDC}")
        DEBUG_FILE = open(DEBUG_FILE_NAME, "w", buffering=1) # buffering=1 flushes after each line
        t = Teffie()
        t.run()
    except KeyboardInterrupt:
        print(f"\n{colors.HEADER}Terminating Teffie{colors.ENDC}")
        DEBUG_FILE.close()
        print(f"\n{colors.HEADER}Teffie terminated{colors.ENDC}")

