# how the heck does async work?

# let's try to create a basic producer / consumer thing in async

# see https://realpython.com/async-io-python/ for later reference

# ultimately, how fast or slow different mthreading solutions are
# depends on how much core utilization we can get on this device
 
import asyncio
import sys
import threading

q = asyncio.Queue()
def record_audio():
    loop = asyncio.get_event_loop()

    def callback():    
        i = 0
        while True:
            q.put(i)
            i += 1

    loop.call_soon(callback,)

    # def callback(indata, frame_count, time_info, status):
    #     pass
    
# this will correspond to our transcription thread / coroutine.
# 
async def consume_audio():
    while True:
        x = await q.get()
        print(x)

# this probably makes sense, we want a "master function" corresponding to
# this thread to do the thread spawning, rather than directly bind the 
# coroutine to the thread

# i'm def thinking in terms of multtiple coroutines per thread. Maybe this is
# not so necessary if you have just one coroutine in the thread?
# def t_master_function(target):
#     # target will be our target function
#     target()
    # asyncio.run(target)

async def main():
    print("Hello")
    # t_record = threading.Thread(target=t_master_function, args=(record_audio,))
    # t_consume = threading.Thread(target=t_master_function, args=(consume_audio,))
    t_record = threading.Thread(target=record_audio)
    # t_consume = threading.Thread(target=consume_audio)


    t_record.start()
    # t_consume.start()    
    

# async def task(name, work_queue):
#     timer = Timer(text=f"Task {name} elapsed time: {{:.1f}}")
#     while not work_queue.empty():
#         delay = await work_queue.get()
#         print(f"Task {name} running")
#         timer.start()
#         await asyncio.sleep(delay)
#         timer.stop()

# async def main():
#     """
#     This is the main entry point for the program
#     """
#     # Create the queue of work
#     work_queue = asyncio.Queue()

#     # Put some work in the queue
#     for work in [15, 10, 5, 2]:
#         await work_queue.put(work)

#     # Run the tasks
#     with Timer(text="\nTotal elapsed time: {:.1f}"):
#         await asyncio.gather(
#             asyncio.create_task(task("One", work_queue)),
#             asyncio.create_task(task("Two", work_queue)),
#         )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nGoodbye')
