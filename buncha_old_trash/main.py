# import threading
# import sounddevice as sd
# import whisper
# import numpy as np
# import asyncio

# # highly based on this example for now:
# # https://github.com/gaborvecsei/whisper-live-transcription/blob/main/standalone-poc/live-transcribe.py

# # architecture: one thread reads from audio input and splices into chunks that
# # can be fed into whisper
# # one thread transcribes with whisper and outputs text, then checks if any 
#     # we can profile and play around with how much it helps to have a different thread read the outputted text,
#     # but unless we have a HUGE keyword list then it shouldn't matter

# # if we feel fancy we could use some nice ui thingy that highlights keywords in outputted text 

# # break recorded audio into chunks of 5 seconds length each
# # EDGE CASE: what if a word spans over multiple chunks?
# # perhaps correct to process two chunks at a time? But then how to avoid duplicating? would be annoying, consider as feat extension
# CHUNK_LENGTH = 5

# running_transcription = ""



# def read_audio():
#     buffer = np.zeros() # this should be a numpy array with length according to fs and CHUNK LENGTH
#     loop = asyncio.get_event_loop()
#     event = asyncio.Event()
#     idx = 0

#     def callback(indata, frame_count, time_info, status):
#         nonlocal idx
#         if status:
#             print(status)
#         remainder = len(buffer) - idx
#         if remainder == 0:
#             loop.call_soon_threadsafe(event.set)
#             raise sd.CallbackStop
#         indata = indata[:remainder]
#         buffer[idx:idx + len(indata)] = indata
#         idx += len(indata)

#     stream = sd.InputStream(callback=callback, dtype=buffer.dtype,
#                             channels=buffer.shape[1], **kwargs)
#     with stream:
#         await event.wait()

#     pass

# def transcribe_audio():
#     return transcription

# def create_response(transcription):
#     pass

# def audio_producer_thread():
#     read_audio()

# def audio_consumer_thread():
#     transcription = transcribe_audio()
#     create_response(transcription)

# if __name__ == "__main__":
#     audio_producer = threading.Thread(target=audio_producer_thread)
#     audio_consumer = threading.Thread(target=audio_consumer_thread)

#     audio_producer.start()
#     audio_consumer.start()

#     try:
#         audio_producer.join()
#         audio_consumer.join()
#     except:
#         pass

# NOTE: you can "free" memory by assigning var = None, which should indicate to the garbage collector
# that mem can be cleaned up

# ideally we could import everything from transformers, but I'm not sure if we can decode numpy arrays
# directly using whisper from transformers
# there may also be a concern of stuff going down and requiring a switch to some other model
import whisper
from transformers import pipeline

import queue 
import threading
import numpy as np
import sounddevice as sd

# import cProfile

from sys import getsizeof

class Teffie:
    # note that setting this too high results in out-of-mem errors. Most user-friendly option is to 
    # calculate based on system RAM / process allowance (can we figure out process allowance?).
    RECORDING_BLOCKSIZE = 1024
    NUM_CHANNELS = 1 # mono, or 2 for stereo; i see no reason to do more for laptop speakers
    NUMPY_DTYPE = np.float32

    class HandleWrapper:
        def __init__(self, handle):
            # print("running ctor")
            self.handle = handle

        def __del__(self):
            # print("running dtor")
            self.handle.truncate(0)
            # print("finished dtor")
        
        def write(self, content):
            self.handle.write(content)

    def __init__(self):
        self.audio_slice_list = [open(f'input_slice_{index}', 'w') for index in range(16)]

        self._input_reader_to_speech_transcriber_queue = queue.Queue()
        self._speech_transcriber_to_text_generator_queue = queue.Queue()
        self._text_generator_to_speech_generator_queue = queue.Queue()
        # https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages for param options
        print("Loading transcription model...")
        self._speech_transcriber = whisper.load_model("tiny")
        print("Done loading transcription model")
        self._text_generator = None
        self._speech_generator = None 
 
    def record_audio(self):
        # buffer doesn't need to be pre-declared like this unless we switch to the async stuff
        # input_buffer = np.empty((self.RECORDING_BLOCKSIZE, 2), dtype=self.NUMPY_DTYPE) 
        recording_stream = sd.InputStream(blocksize=0, latency='low', dtype=self.NUMPY_DTYPE)
        recording_stream.start()
        try:
            while True:
                # hypothetically this causes a problem if A. we made the operation after the read more expensive or 2 (much more likely).
                # self.q blocks because the other thread is using it, which will get worse the larger the queue gets
                # the problem is that we will skip audio input while we wait on these other operations 
                
                # let's just go with this for now and see what we can do to make sure we keep reading throughout the program
                # lifetime - likely we need to utilize callbacks
                read_buffer, overflowed = recording_stream.read(self.RECORDING_BLOCKSIZE)
                # I don't think we do anything with overflowed
                
                # if status:
                self._input_reader_to_speech_transcriber_queue.put(read_buffer.copy())
                
                if overflowed:
                    print("Hey we overflowed, but who cares")
                # else: 
                #     print("Something went wrong with recording_stream.read operation")
                #     recording_stream.stop()
                #     break
                # print(self._input_reader_to_speech_transcriber_queue.qsize())
        except KeyboardInterrupt:
            recording_stream.stop()

    # TODO: figure out how to staple audio together based on "volume" of speech
    #       see noisereduce package for more, maybe?
    def transcribe_audio(self):
        while True:
            next_input = self._input_reader_to_speech_transcriber_queue.get()
            print(next_input)
            print("Starting transcription...")
            # we may need to use lower level functions here. transcribe uses a 30 second window
            # which is way longer than the input we are passing in anyway
            transcription = self._speech_transcriber.transcribe(next_input)
            print("Finished transcription")
            self._speech_transcriber_to_text_generator_queue.put(transcription)
            print(self._speech_transcriber_to_text_generator_queue.qsize())

    def generate_text(self):
        while True:
            # for now, just dump this
            self._speech_transcriber_to_text_generator_queue.get()

    def run(self):
        _recording_thread = threading.Thread(target=self.record_audio)
        _transcribing_thread = threading.Thread(target=self.transcribe_audio)
        _generating_thread = threading.Thread(target=self.generate_text)

        _recording_thread.start()
        _transcribing_thread.start()
        _generating_thread.start()

if __name__ == "__main__":
    try:
        print("\nHello")
        t = Teffie()
        t.run()
        # cProfile.run('t.run()')
    except KeyboardInterrupt:
        print("\nGoodbye")

