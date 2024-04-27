# may want to rename this "teffie_logger" or similar
# to avoid confusion with torch logging module
import yaml

with open("_config.yml", 'r') as _config:
    _config_data = yaml.safe_load(_config)

DEBUG_ENABLED = _config_data["debug"]["is_enabled"]
DEBUG_SHOULD_WRITE_TO_FILE = _config_data["debug"]["should_write_to_file"]
DEBUG_FILENAME = _config_data["debug"]["filename"]

if DEBUG_SHOULD_WRITE_TO_FILE:
    # buffering=1 flushes after each line to rather than at end of execution,
    # ensuring output is actually preserved
    DEBUG_FILE = open(DEBUG_FILENAME, "w", buffering=1) 


# reference https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
class colors:
    INFO           = '\033[92m'
    DEBUG_INFO     = '\033[32m'
    WARN           = '\033[93m'
    DEBUG_WARN     = '\033[33m'
    ERROR          = '\033[91m'
    DEBUG_ERROR    = '\033[31m'
    CRITICAL       = '\033[95m'
    DEBUG_CRITICAL = '\033[35m'
    ENDC           = '\033[0m'

# NOTE: does string formatting allow us to print non-string objects?

# for stuff that should always be printed for end user
def LOG_INFO(string):
    print(f"{colors.INFO}{string}{colors.ENDC}")

def LOG_WARN(string):
    print(f"{colors.WARN}{string}{colors.ENDC}")

def LOG_ERROR(string):
    print(f"{colors.ERROR}{string}{colors.ENDC}")

def LOG_CRITICAL(string):
    print(f"{colors.CRITICAL}{string}{colors.ENDC}")

# for stuff that should only be printed for developers
def LOG_DEBUG_INFO(string):
    if DEBUG_ENABLED:
        print(f"{colors.DEBUG_INFO}{string}{colors.ENDC}")
        if DEBUG_SHOULD_WRITE_TO_FILE:
            DEBUG_FILE.write(f"{string}\n")

def LOG_DEBUG_WARN(string):
    if DEBUG_ENABLED:
        print(f"{colors.DEBUG_WARN}{string}{colors.ENDC}")
        if DEBUG_SHOULD_WRITE_TO_FILE:
            DEBUG_FILE.write(f"{string}\n")

def LOG_DEBUG_ERROR(string):
    if DEBUG_ENABLED:
        print(f"{colors.DEBUG_ERROR}{string}{colors.ENDC}")
        if DEBUG_SHOULD_WRITE_TO_FILE:
            DEBUG_FILE.write(f"{string}\n")

def LOG_DEBUG_CRITICAL(string):
    if DEBUG_ENABLED:
        print(f"{colors.DEBUG_CRITICAL}{string}{colors.ENDC}")
        if DEBUG_SHOULD_WRITE_TO_FILE:
            DEBUG_FILE.write(f"{string}\n")
    
def _close_debug_file():
    DEBUG_FILE.close()

