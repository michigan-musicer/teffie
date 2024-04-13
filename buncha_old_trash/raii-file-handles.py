import sys

class RAIIFileHandle:
    def __init__(self, handle):
        print("running ctor")
        self.handle = handle

    def write(self, content):
        self.handle.write(content)

    def __del__(self):
        print("running dtor")
        # this technically works to clear the file, but i'd like to reserve space in the file
        # for the input if possible
        self.handle.truncate(0)
        print("finished dtor")

def main():
    audio_slice_list = [open(f'input_slice_{index}', 'w') for index in range(16)]
    for i in range(len(audio_slice_list)):
        file = RAIIFileHandle(audio_slice_list[i])
        file.write(f'this is file {i}')


if __name__ == "__main__":
    main()
