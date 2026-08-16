COLORS_DATA = {
    "RED": '\033[91m',
    "YELLOW": '\033[93m',
    "GREEN": '\033[92m',
    "BLUE": '\033[94m',
    "MAGENTA":  '\033[95m',
    "CYAN":  '\033[96m',
    "WHITE": '\033[97m',
    "ENDC": '\033[0m',  # Reset code
}

def colorize(text, color):
    return f'{COLORS_DATA.get(color, "")}{text}{COLORS_DATA.get("ENDC")}'
