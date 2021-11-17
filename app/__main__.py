import sys
import os

if __name__ == '__main__':
    print(sys.argv[1])
    print(os.path.join(os.path.join(os.path.dirname(__file__), ".."), sys.argv[1]))
