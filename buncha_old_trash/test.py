"""An example for using a stream in an asyncio coroutine.

This example shows how to create a stream in a coroutine and how to wait for
the completion of the stream.

You need Python 3.7 or newer to run this.

"""
import asyncio
import sys

import numpy as np
import sounddevice as sd


async def record_buffer(buffer, **kwargs):
    loop = asyncio.get_event_loop()
    event = asyncio.Event()
    idx = 0

    def callback(indata, frame_count, time_info, status):
        nonlocal idx
        if status:
            print(status)
        remainder = len(buffer) - idx
        if remainder == 0:
            loop.call_soon_threadsafe(event.set)
            raise sd.CallbackStop
        indata = indata[:remainder]
        buffer[idx:idx + len(indata)] = indata
        idx += len(indata)

    stream = sd.InputStream(callback=callback, dtype=buffer.dtype,
                            channels=buffer.shape[1], **kwargs)
    with stream:
        await event.wait()


async def play_buffer(buffer, **kwargs):
    loop = asyncio.get_event_loop()
    event = asyncio.Event()
    idx = 0

    def callback(outdata, frame_count, time_info, status):
        nonlocal idx
        if status:
            print(status)
        remainder = len(buffer) - idx
        if remainder == 0:
            loop.call_soon_threadsafe(event.set)
            raise sd.CallbackStop
        valid_frames = frame_count if remainder >= frame_count else remainder
        outdata[:valid_frames] = buffer[idx:idx + valid_frames]
        outdata[valid_frames:] = 0
        idx += valid_frames

    stream = sd.OutputStream(callback=callback, dtype=buffer.dtype,
                             channels=buffer.shape[1], **kwargs)
    with stream:
        await event.wait()


async def main(frames=150_000, channels=1, dtype='float32', **kwargs):
    buffer = np.empty((frames, channels), dtype=dtype)
    print('recording buffer ...')
    await record_buffer(buffer, **kwargs)
    print('playing buffer ...')
    await play_buffer(buffer, **kwargs)
    print('done')


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')

# # import asyncio
# # import sys

# # import numpy as np
# # import sounddevice as sd

# # import queue
# # q = queue.Queue()

# # FRAMES = 150000
# # CHANNELS = 2
# # D_TYPE = 'float32'

# # # we should work on getting the callback architecture working because otherwise 
# # # we will have "jumps" in the read audio corresponding to all the operations done
# # # between input_stream.read(...) calls.
# # # May not be worrisome now, but may be a problem later.
# # # (pre-optimization is the root of all evil and I am the devil)
# # # (I mean also i just feel like i should be able to get this working lol)
# # def callback(indata, FRAMES, time, status):
# #     if status:
# #         print(status)
# #     q.put(indata.copy())
# #     # x = indata
# #     # print(indata)

# # input_stream = sd.InputStream(channels=CHANNELS, latency='low', callback=None)
# # with input_stream:
# #     input_stream.start()
# #     should_keep_streaming = True
# #     input_buffer = np.empty((FRAMES, CHANNELS), dtype=D_TYPE)
# #     while should_keep_streaming:
# #         # print(q.get())
# #         # input_buffer = input_stream.read(FRAMES)
# #         print("here")
# #         print(q.get())

# #         # should_keep_streaming = False
# #     input_stream.stop()

# import queue 
# import threading

# q = queue.Queue()
# def record_audio():
#     i = 0
#     while True:
#         q.put(i)
#         i += 1

# def consume_audio():
#     while True:
#         print(q.get())

# def main():
#     print("\nHello")
#     _recording_thread = threading.Thread(target=record_audio)
#     _consuming_thread = threading.Thread(target=consume_audio)

#     _recording_thread.start()
#     _consuming_thread.start()

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\nGoodbye")
