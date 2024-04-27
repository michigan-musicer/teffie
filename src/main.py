import torch
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer, VitsModel, AutoTokenizer, QuantoConfig
from transformers.utils import logging
# mayyyyyybe we should wrap this in a class, but making the import statement look nice is not really a priority for me
# especially since logging shouldn't be added to beyond this point
from logger import LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL, LOG_DEBUG_INFO, LOG_DEBUG_WARN, LOG_DEBUG_ERROR, LOG_DEBUG_CRITICAL, _close_debug_file

import numpy as np
import sounddevice as sd
import threading

import time
import yaml

logging.set_verbosity_error()
torch.set_default_device("cuda")

with open("_config.yml", 'r') as _config:
    _config_data = yaml.safe_load(_config)

class Teffie:
    sd.default.samplerate = _config_data["audio"]["sample_rate"]
    _FRAMES_PER_SECOND = _config_data["audio"]["frames_per_second"]
    _SECONDS_PER_CHUNK = _config_data["audio"]["seconds_per_chunk"]
    FRAMES_PER_CHUNK = int(_FRAMES_PER_SECOND * _SECONDS_PER_CHUNK)
    sd.default.channels = (_config_data["audio"]["num_input_channels"], _config_data["audio"]["num_output_channels"])
    NUM_INPUT_CHANNELS = sd.default.channels[0]
    NUM_OUTPUT_CHANNELS = sd.default.channels[1]
    match _config_data["audio"]["numpy_dtype"]:
        case "float32":
            NUMPY_DTYPE = np.float32
        case _:
            NUMPY_DTYPE = np.float32

    def __init__(self):
        LOG_DEBUG_INFO(f"FRAMES_PER_CHUNK is {self.FRAMES_PER_CHUNK}")
        LOG_DEBUG_INFO(f"NUM_INPUT_CHANNELS is {self.NUM_INPUT_CHANNELS}")
        LOG_DEBUG_INFO(f"NUM_OUTPUT_CHANNELS is {self.NUM_OUTPUT_CHANNELS}")
        LOG_DEBUG_INFO(f"NUMPY_DTYPE is {self.NUMPY_DTYPE}")

        LOG_INFO("Loading audio transcription model...")
        self._speech_transcriber = whisper.load_model("base.en")
        LOG_INFO("Done loading audio transcription model")
        
        # NOTE: these load from the internet, which is not ideal with our school's crappy wifi
        # preferably to load from cache
        LOG_INFO("Loading text generation model...")
        quantization_config = QuantoConfig(weights="int8")
        # is there a better way to load the model onto GPU? compare and contrast perf using to("cuda") vs device_map="cuda:0"
        self._text_generator = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", quantization_config=quantization_config, torch_dtype="auto").to("cuda")
        LOG_INFO("Done loading text generation model")
        self._text_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

        LOG_INFO("Loading speech generation model...")
        self._speech_generator = VitsModel.from_pretrained("facebook/mms-tts-eng") 
        self._speech_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        LOG_INFO("Done loading speech generation model")

        self._should_stop_recording = False

    # NOTE: this is very hacky and should be swapped out when we get a chance
    # at the very least use a coroutine man!
    # NOTE: is confusing if we go 30 seconds because then we need to hit enter to stop this thread
    # before hitting enter again to start recording.
    # We just need to rearchitect this listener.
    def _keyboard_listener(self):
        time.sleep(0.5)
        input()
        LOG_DEBUG_INFO("Recording stopped by keyboard interrupt")
        self._should_stop_recording = True
    
    def record_audio(self):
        input("Press ENTER to start recording.")
        LOG_INFO("Starting recording...")
        time.sleep(0.2)
        LOG_CRITICAL("Press ENTER to stop recording. Recording will automatically end at 30 seconds.")
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
        LOG_INFO("Finished recording")

        return buffer

    def transcribe_audio(self, next_input):
        # applies all transformations to np array necessary to pass into whisper model 
        # referencing https://github.com/openai/whisper/discussions/450
        def transform_buffer(buffer):
            # x = torch.from_numpy(buffer.flatten() / 32768.0).float()
            # print(x.shape)
            # print(whisper.pad_or_trim(x).shape)
            # raise Exception("no explanation, screw you")
            return whisper.pad_or_trim(torch.from_numpy(buffer.flatten()).float())
        decoding_options = whisper.DecodingOptions(language="en")
        LOG_INFO("Starting transcription...")
        start = time.time()
        next_input = transform_buffer(next_input)
        mel = whisper.log_mel_spectrogram(next_input).to(self._speech_transcriber.device)
        transcription = whisper.decode(self._speech_transcriber, mel, decoding_options).text
        LOG_INFO(f"Transcription is \"{transcription}\"")
        end = time.time()
        LOG_DEBUG_INFO(f"Time elapsed during response generation: {end - start}")
        LOG_INFO("Finished transcription")
        return transcription

    def _truncate_response(self, response):
        LOG_DEBUG_INFO(f"Original response:\n{response}")
        # step 1: truncate up to "Answer:"
        answer_start_index = response.index("Assistant: ")
        response = response[answer_start_index + len("Assistant: "):]
        # step 1.5: remove leading and trailing whitespace
        response = response.strip()

        # step 2: use only the first line of the response
        # NOTE: we get a lot of dead responses if we do this bc edge cases. This should hopefully go away with fine-tuning.
        last_newline_index = response.find('\n') 
        last_newline_index = len(response) if last_newline_index == -1 else last_newline_index
        response = response[:last_newline_index]
        # step 3: only include up to the last stopping character (comma, period, q-mark, excl-mark) and make that stopping point a period
        last_puncutation_mark_index = max(response.rfind(mark) for mark in ['.', ',', '?', '!']) 
        last_puncutation_mark_index = len(response) if last_puncutation_mark_index == -1 else last_puncutation_mark_index
        # AFAIK Python doesn't let you change one character of a string, so this is a workaround 
        response = response[:last_puncutation_mark_index] + '.'
        return response

    def generate_response(self, transcription):
        LOG_INFO("Starting response generation...")
        start = time.time()
        inputs = self._text_tokenizer(f'''The assistant obeys the user completely and gives helpful, detailed answers to the user's questions. \nUser: reply to the following introduction as a different character: {transcription} \nAssistant: ''', return_tensors="pt", return_attention_mask=False)

        outputs = self._text_generator.generate(**inputs, max_new_tokens=64, do_sample="False")

        response = self._text_tokenizer.batch_decode(outputs)[0]
        response = self._truncate_response(response)
        LOG_INFO(f"Generated response: \"{response}\"")
        end = time.time()
        LOG_DEBUG_INFO(f"Time elapsed during response generation: {end - start}")
        LOG_INFO("Finished response generation")
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
                LOG_WARN(f"callback status is {status}")
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
        LOG_CRITICAL("Running Teffie...")
        t = Teffie()
        t.run()
    except KeyboardInterrupt:
        LOG_CRITICAL("Terminating Teffie...")
        _close_debug_file()
        LOG_CRITICAL("Teffie terminated")

