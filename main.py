from transformers import pipeline

import os
import queue 
import threading
import numpy as np
import sounddevice as sd

DEBUG = True
def debug_print(string):
    if DEBUG:
        print(string)

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
    FRAMES_PER_CHUNK = int(sd.query_devices(sd.default.device)["default_samplerate"])
    NUM_CHANNELS = int(sd.query_devices(sd.default.device)["max_input_channels"]) 
    NUMPY_DTYPE = np.float32
    NUM_HANDLES = 16

    class HandleQueue:
        class HandleWrapper:
            def __init__(self, handle):
                # print("running ctor")
                self.handle = handle

            def __del__(self):
                # print("running dtor")
                self.handle.truncate(0)
                # print("finished dtor")
            
            # requires content to be ndarray type.
            def write(self, content):
                content.tofile(self.handle)
                # self.handle.write(content) # this is the naive way
        
            def clear(self):
                self.handle.truncate(0)

        # root_path is the directory which will contain all the files controlled by HandleQueue.
        def __init__(self, num_handles, root_path):
            self._num_handles = num_handles 

            if not os.path.exists(root_path):
                print(f"{colors.WARNING}directory {root_path} does not exist. Creating {root_path}...{colors.ENDC}")
                os.mkdir(f'./{root_path}')
            # Queue is impl'd by ring buffer.
            self._handle_ring_buffer = [self.HandleWrapper(open(f'input_slice_{index}', 'w')) for index in range(self._num_handles)]
            self._ring_start = -1 # begins as invalid
            self._ring_end = 0
            self._size = 0

            self._cv = threading.Condition()
        
        def size(self):
            return self._size
        
        def empty(self):
            return self._size == 0
        
        def has_item(self):
            return self._size > 0

        # fetches and returns data from first written-to file 
        def get(self):
            debug_print(f"{colors.WARNING}Popping from HandleQueue{colors.ENDC}")
            self._cv.acquire()
            debug_print(f"{colors.WARNING}Condition lock acquired by HandleQueue.get call{colors.ENDC}")
            while not self.has_item():
                debug_print(f"{colors.WARNING}HandleQueue empty, waiting for item{colors.ENDC}")
                self._cv.wait()
            debug_print(f"{colors.WARNING}HandleQueue pop no longer waiting, HandleQueue size is {self.size()}{colors.ENDC}")
            return_buffer = np.fromfile(self._handle_ring_buffer[self._ring_start])
            self._handle_ring_buffer[self._ring_start].clear()
            self._ring_start = (self._ring_start + 1) % self._num_handles
            self._size -= 1 
            debug_print(f"{colors.WARNING}item popped from queue in HandleQueue.get call{colors.ENDC}")
            self._cv.release()
            debug_print(f"{colors.WARNING}Condition lock released by HandleQueue.get call{colors.ENDC}")
            return return_buffer

        # writes to end position
        def put(self, data):
            debug_print(f"{colors.WARNING}Putting onto HandleQueue{colors.ENDC}")
            self._cv.acquire()
            debug_print(f"{colors.WARNING}Condition lock acquired by HandleQueue.put call{colors.ENDC}")
            if self._size == self._num_handles:
                raise Exception(f"{colors.ERROR}attempted to write to full HandleQueue object{colors.ENDC}")
            self._handle_ring_buffer[self._ring_end].write(data)
            self._ring_end = (self._ring_end + 1) % self._num_handles
            self._size += 1
            debug_print(f"{colors.WARNING}item added to queue in HandleQueue.put call{colors.ENDC}")
            self._cv.notify()
            debug_print(f"{colors.WARNING}Notify sent from HandleQueue.put call{colors.ENDC}")
            self._cv.release()
            debug_print(f"{colors.WARNING}Condition lock released by HandleQueue.put call{colors.ENDC}")


    def __init__(self):
        debug_print(f"{colors.WARNING}FRAMES_PER_CHUNK is {self.FRAMES_PER_CHUNK}{colors.ENDC}")
        debug_print(f"{colors.WARNING}NUM_CHANNELS is {self.NUM_CHANNELS}{colors.ENDC}")
        debug_print(f"{colors.WARNING}NUMPY_DTYPE is {self.NUMPY_DTYPE}{colors.ENDC}")
        debug_print(f"{colors.WARNING}NUM_HANDLES is {self.NUM_HANDLES}{colors.ENDC}")
        
        self._input_reader_to_speech_transcriber_queue = self.HandleQueue(self.NUM_HANDLES, "audio_slices")
        # we may need to use the handle queue for these other two queues? depends how much accumulation we have
        self._speech_transcriber_to_text_generator_queue = queue.Queue()
        self._text_generator_to_speech_generator_queue = queue.Queue()

        print(f"{colors.OKCYAN}Loading audio transcription pipeline...{colors.ENDC}")
        self._speech_transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
        print(f"{colors.OKCYAN}Done loading audio transcription pipeline{colors.ENDC}")
        self._text_generator = None
        self._speech_generator = None 

    def record_audio(self):
        buffer = np.empty((self.FRAMES_PER_CHUNK, self.NUM_CHANNELS), dtype=self.NUMPY_DTYPE)
        def refresh_buffer():
            nonlocal buffer
            buffer = np.empty((self.FRAMES_PER_CHUNK, self.NUM_CHANNELS), dtype=self.NUMPY_DTYPE)
            
        def detect_silence(buffer):
            debug_print(f"{colors.WARNING}beginning silence detection{colors.ENDC}")
            # debug_print(f"{colors.FAIL}using placeholder silence detection to never enqueue buffers for transcription{colors.ENDC}")
            # return False, 0, 0 # uncomment this to never enqueue blocks for transcription
            debug_print(f"{colors.FAIL}using placeholder silence detection to always enqueue buffers for transcription{colors.ENDC}")
            return True, buffer.size, buffer.size # uncomment this to always enqueue blocks for transcription

        idx = 0
        current_block_is_done = False
        buffer_locked_by_main = False
        # this is copied directly from the asyncio callback example for sounddevice; as there's no
        # need to adapt this any more specifically to our needs. May change as I learn mroe about the problem.
        def callback(indata, frame_count, time_info, status):
            nonlocal buffer, idx
            nonlocal current_block_is_done
            nonlocal buffer_locked_by_main
            
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
            cumulative_buffer = np.empty((0, self.NUM_CHANNELS), dtype=self.NUMPY_DTYPE)
            # what's the simplest queue we can use for this? we really just need a FIFO data structure
            processed_audio_queue = queue.SimpleQueue()
            while True:
                if current_block_is_done:
                    # we need to lock buffer so that callback isn't called and overwrites the buffer
                    # since there's only two concurrent thingies using buffer, we can do a simple boolean lock i think
                    # I would prefer a solution that tells callback to pause until done
                    buffer_locked_by_main = True
                    debug_print(f"{colors.WARNING}buffer_locked_by_main set to True, placing into processed_audio_queue{colors.ENDC}")
                    processed_audio_queue.put(buffer)
                    debug_print(f"{colors.WARNING}refreshing buffer{colors.ENDC}")
                    refresh_buffer()
                    debug_print(f"{colors.WARNING}buffer refreshed, about to release lock{colors.ENDC}")
                    buffer_locked_by_main = False
                    current_block_is_done = False
                    debug_print(f"{colors.WARNING}buffer_locked_by_main set to False, allowing callback to resume{colors.ENDC}")

                    next_buffer = processed_audio_queue.get()
                    debug_print(f"{colors.WARNING}acquired next buffer to process from {colors.ENDC}")
                    has_silence, silence_start_index, silence_end_index = detect_silence(next_buffer)
                    # check dimension here
                    if has_silence and cumulative_buffer.size > 0:
                        # append up to silence, push to queue, and then restart cumulative buffer from end of silence.
                        cumulative_buffer = np.append(cumulative_buffer, next_buffer[:silence_start_index], axis=0)
                        self._input_reader_to_speech_transcriber_queue.put(cumulative_buffer)
                        cumulative_buffer = next_buffer[silence_end_index:]
                    else:
                        # add all of the last buffer to cumulative buffer.
                        cumulative_buffer = np.append(cumulative_buffer, next_buffer, axis=0)
                    

                    
                

    def transcribe_audio(self):
        while True:
            next_input = self._input_reader_to_speech_transcriber_queue.get()
            print(next_input)
            print("Starting transcription...")
            # NOTE: replacement starts here
            # we may need to use lower level functions here. transcribe uses a 30 second window
            # which is way longer than the input we are passing in anyway
            transcription = self._speech_transcriber.transcribe(next_input)
            print("Finished transcription")
            self._speech_transcriber_to_text_generator_queue.put(transcription)
            print(self._speech_transcriber_to_text_generator_queue.qsize())

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
        t = Teffie()
        t.run()
        # cProfile.run('t.run()')
    except KeyboardInterrupt:
        print(f"\n{colors.HEADER}Terminated{colors.ENDC}")

