import whisper
# from transformers import pipeline
import torch

import os
import queue 
import threading
import numpy as np
import sounddevice as sd
import asyncio

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
    # sample rate can be thought of as frames per second, so using this value
    # by itself is basically saying "each chunk is 1 second" 
    # We convert to int to avoid type errors when creating numpy arrays
    # _FRAMES_PER_SECOND = int(sd.query_devices(sd.default.device)["default_samplerate"])
    # 16000 is the sample rate for whisper and must be followed by low-level code
    sd.default.samplerate = 16000
    _FRAMES_PER_SECOND = 16000
    # fiddle with this to change chunk size
    _SECONDS_PER_CHUNK = 0.5
    FRAMES_PER_CHUNK = _FRAMES_PER_SECOND * _SECONDS_PER_CHUNK
    # forcing 1 channel minimizes data size, which will be helpful for speed
    sd.default.channels = 1
    # sd.default.channels is a tuple, so just grab the first and second each
    NUM_INPUT_CHANNELS = sd.default.channels[0]
    NUM_OUTPUT_CHANNELS = sd.default.channels[1]
    # NUM_INPUT_CHANNELS = int(sd.query_devices(sd.default.device)["max_input_channels"]) 
    NUMPY_DTYPE = np.float32
    NUM_HANDLES = 1024

    class HandleQueue:
        class HandleWrapper:
            def __init__(self, handle, filename):
                # print("running ctor")
                self._handle = handle
                self._filename = filename

            def __del__(self):
                # print("running dtor")
                self._handle.truncate(0)
                # print("finished dtor")

            def get_handle(self):
                return self._handle
            
            def get_filename(self):
                return self._filename

            # requires content to be ndarray type.
            def write(self, content):
                content.tofile(self._handle)
                # self._handle.write(content) # this is the naive way
        
            def clear(self):
                self._handle.truncate(0)

        # root_path is the directory which will contain all the files controlled by HandleQueue.
        def __init__(self, num_handles, root_path):
            self._num_handles = num_handles 

            if not os.path.exists(root_path):
                print(f"{colors.WARNING}directory {root_path} does not exist. Creating {root_path}...{colors.ENDC}")
                os.mkdir(f'./{root_path}')
            # File handles are enqueued onto an async-friendly queue.
            self._queue = queue.Queue() 
        
        def size(self):
            return self._size
        
        def empty(self):
            return self._size == 0
        
        def has_item(self):
            return self._size > 0

        # fetches and returns data from first written-to file 
        def get(self):
            debug_print(f"{colors.WARNING}Popping from HandleQueue{colors.ENDC}")
            debug_print(f"{colors.WARNING}About to begin waiting for item in HandleQueue pop{colors.ENDC}")
            while not self.has_item():
                debug_print(f"{colors.FAIL}Busy waiting in get call in HandleQueue{colors.ENDC}")
                pass    
            debug_print(f"{colors.WARNING}HandleQueue pop not waiting, HandleQueue size is {self.size()}{colors.ENDC}")
            return_buffer = np.fromfile(self._handle_ring_buffer[self._ring_start].get_filename())
            self._handle_ring_buffer[self._ring_start].clear()
            self._ring_start = (self._ring_start + 1) % self._num_handles
            self._size -= 1 
            debug_print(f"{colors.WARNING}item popped from queue in HandleQueue.get call{colors.ENDC}")
            return return_buffer

        # writes to end position
        def put(self, data):
            debug_print(f"{colors.WARNING}Putting onto HandleQueue{colors.ENDC}")
            if self._size == self._num_handles:
                raise Exception(f"{colors.ERROR}attempted to write to full HandleQueue object{colors.ENDC}")
            self._handle_ring_buffer[self._ring_end].write(data)
            self._ring_end = (self._ring_end + 1) % self._num_handles
            self._size += 1
            debug_print(f"{colors.WARNING}item added to queue in HandleQueue.put call{colors.ENDC}")

    def __init__(self):
        debug_print(f"{colors.WARNING}FRAMES_PER_CHUNK is {self.FRAMES_PER_CHUNK}{colors.ENDC}")
        debug_print(f"{colors.WARNING}NUM_INPUT_CHANNELS is {self.NUM_INPUT_CHANNELS}{colors.ENDC}")
        debug_print(f"{colors.WARNING}NUM_OUTPUT_CHANNELS is {self.NUM_OUTPUT_CHANNELS}{colors.ENDC}")
        debug_print(f"{colors.WARNING}NUMPY_DTYPE is {self.NUMPY_DTYPE}{colors.ENDC}")
        debug_print(f"{colors.WARNING}NUM_HANDLES is {self.NUM_HANDLES}{colors.ENDC}")
        
        # TODO: convert this to two queues: we fill disk in record_audio, then fill an async queue 
        # or we change the setup of handleQueue so that we have one queue of available handles and one queue of actual
        # information that is exposed to consumers
        # basically, producer => files, consumer <= async queue (what a weird setup....)
        # or just use event.wait here as well lol
        self._input_reader_to_speech_transcriber_queue = self.HandleQueue(self.NUM_HANDLES, "audio_slices")
        # we may need to use the handle queue for these other two queues? depends how much accumulation we have
        self._speech_transcriber_to_text_generator_queue = queue.Queue()
        self._text_generator_to_speech_generator_queue = queue.Queue()

        print(f"{colors.OKCYAN}Loading audio transcription model...{colors.ENDC}")
        # self._speech_transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
        self._speech_transcriber = whisper.load_model("tiny.en")
        print(f"{colors.OKCYAN}Done loading audio transcription model{colors.ENDC}")
        self._text_generator = None
        self._speech_generator = None 

    def record_audio(self):
        def refresh_buffer():
            nonlocal buffer
            buffer = np.empty((self.FRAMES_PER_CHUNK, self.NUM_INPUT_CHANNELS), dtype=self.NUMPY_DTYPE)
            
        def detect_silence(buffer):
            debug_print(f"{colors.WARNING}beginning silence detection{colors.ENDC}")
            # debug_print(f"{colors.FAIL}using placeholder silence detection to never enqueue buffers for transcription{colors.ENDC}")
            # return False, 0, 0 # uncomment this to never enqueue blocks for transcription
            debug_print(f"{colors.FAIL}using placeholder silence detection to always enqueue buffers for transcription{colors.ENDC}")
            return True, buffer.size, buffer.size # uncomment this to always enqueue blocks for transcription

        buffer = np.empty((self.FRAMES_PER_CHUNK, self.NUM_INPUT_CHANNELS), dtype=self.NUMPY_DTYPE)
        idx = 0
        current_block_is_done = False
        buffer_locked_by_main = False
        # this is copied directly from the asyncio callback example for sounddevice; as there's no
        # need to adapt this any more specifically to our needs. May change as I learn mroe about the problem.
        def callback(indata, frame_count, time_info, status):
            nonlocal buffer, idx
            nonlocal current_block_is_done
            nonlocal buffer_locked_by_main
            
            debug_print(f"{colors.FAIL}in callback{colors.ENDC}")
            if status:
                print(status)
            if buffer_locked_by_main:
                debug_print(f"{colors.WARNING}Buffer lock should be held by main. Actual value: {buffer_locked_by_main}{colors.ENDC}")
            if not buffer_locked_by_main:
                remainder = len(buffer) - idx
                if remainder == 0:
                    debug_print(f"{colors.WARNING}current_block_is_done set to True in callback{colors.ENDC}")
                    current_block_is_done = True
                    idx = 0 

                    # don't raise sd.CallbackStop because this stops callback from being called ever? 
                    # and we want callback to keep calling once we finish processing the rest of the stuff
                    return
                # NOTE: this line may have to be changed to avoid truncating data. Currently, this discards "extra"
                # data that doesn't fit within 1 total second.
                # maybe do the silence checking and shifting here? 
                indata = indata[:remainder]
                buffer[idx:idx + len(indata)] = indata
                idx += len(indata)

        input_stream = sd.InputStream(callback=callback, latency='low', dtype=self.NUMPY_DTYPE)
        with input_stream:
            cumulative_buffer = np.empty((0, self.NUM_INPUT_CHANNELS), dtype=self.NUMPY_DTYPE)
            # what's the simplest queue we can use for this? we really just need a FIFO data structure
            processed_audio_queue = queue.SimpleQueue()
            while True:
                debug_print(f"{colors.FAIL}Busy waiting in record_audio outside of callback{colors.ENDC}")
                # TODO: stop busy waiting! Use a mutex or find bindings to C functions that do what you want!
                if current_block_is_done:
                # I would prefer a solution that tells fn not to go to callback until done, but I don't know if this is possible
                # in the sounddevice library
                    buffer_locked_by_main = True
                    debug_print(f"{colors.WARNING}buffer_locked_by_main set to True, placing into processed_audio_queue{colors.ENDC}")
                    processed_audio_queue.put(buffer)
                    debug_print(f"{colors.WARNING}refreshing buffer{colors.ENDC}")
                    refresh_buffer()
                    debug_print(f"{colors.WARNING}buffer refreshed, about to release lock{colors.ENDC}")
                    buffer_locked_by_main = False
                    current_block_is_done.clear()
                    debug_print(f"{colors.WARNING}buffer_locked_by_main set to False, allowing callback to resume{colors.ENDC}")

                    next_buffer = processed_audio_queue.get()
                    debug_print(f"{colors.WARNING}acquired next buffer to process from {colors.ENDC}")
                    has_silence, silence_start_index, silence_end_index = detect_silence(next_buffer)
                    if has_silence and cumulative_buffer.size > 0:
                        # append up to silence, push to queue, and then restart cumulative buffer from end of silence.
                        cumulative_buffer = np.append(cumulative_buffer, next_buffer[:silence_start_index], axis=0)
                        self._input_reader_to_speech_transcriber_queue.put(cumulative_buffer)
                        cumulative_buffer = next_buffer[silence_end_index:]
                        debug_print(f"{colors.WARNING}buffer pushed to queue. Current number of file handles in use: {self._input_reader_to_speech_transcriber_queue.size()}{colors.ENDC}")
                    else:
                        # add all of the last buffer to cumulative buffer.
                        cumulative_buffer = np.append(cumulative_buffer, next_buffer, axis=0)
                    if self._input_reader_to_speech_transcriber_queue.size() > 16:
                        debug_print(f"{colors.FAIL}stopping early for profiling convenience{colors.ENDC}")
                        return
                    

                    

    def transcribe_audio(self):
        # applies all transformations to np array necessary to pass into whisper model 
        # referencing https://github.com/openai/whisper/discussions/450
        # my guess is for silence detection, we may repeat some work
        def transform_buffer(buffer):
            x = torch.from_numpy(buffer.flatten() / 32768.0).float()
            print(x.shape)
            print(whisper.pad_or_trim(x).shape)
            return whisper.pad_or_trim(torch.from_numpy(buffer.flatten() / 32768.0).float())

        decoding_options = whisper.DecodingOptions(language="en")
        while True:
            next_input = self._input_reader_to_speech_transcriber_queue.get()
            # print(next_input)
            if next_input.size == 0:
                debug_print(f"{colors.WARNING}input size is 0, skipping transcription{colors.ENDC}")
                continue
            print("Starting transcription...")
            # NOTE: replacement starts here
            # we may need to use lower level functions here. transcribe uses a 30 second window
            # which is way longer than the input we are passing in anyway
            # transcription = self._speech_transcriber(next_input)
            # next_input = torch.from_numpy(next_input).float()
            # next_input = whisper.pad_or_trim(next_input)
            next_input = transform_buffer(next_input)
            mel = whisper.log_mel_spectrogram(next_input).to(self._speech_transcriber.device)
            transcription = whisper.decode(self._speech_transcriber, mel, decoding_options)
            print(transcription)
            # transcription = self._speech_transcriber.transcribe(next_input)
            print("Finished transcription")
            # self._speech_transcriber_to_text_generator_queue.put(transcription)
            # print(self._speech_transcriber_to_text_generator_queue.qsize())

    def generate_text(self):
        pass

    def run(self):
        _recording_thread = threading.Thread(target=self.record_audio)
        _transcribing_thread = threading.Thread(target=self.transcribe_audio)
        _generating_thread = threading.Thread(target=self.generate_text)

        _recording_thread.start()
        _transcribing_thread.start()
        _generating_thread.start()

if __name__ == "__main__":
    try:
        print(f"{colors.HEADER}Running teffie...{colors.ENDC}")
        DEBUG_FILE = open(DEBUG_FILE_NAME, "w", buffering=1) # buffering=1 flushes after each line
        t = Teffie()
        t.run()
        # cProfile.run('t.run()')
    except KeyboardInterrupt:
        print(f"\n{colors.HEADER}Termination begun{colors.ENDC}")
        DEBUG_FILE.close()
        print(f"\n{colors.HEADER}Termination finished{colors.ENDC}")

