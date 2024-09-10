

from .app.meetingsys import start_meeting
from .config import load_config

def start_app():
    args = load_config()
    start_meeting(args)

if __name__ == '__main__':
    args = load_config()
    start_meeting(args)
