
import sys
import os
import time

# Keep stdin open with dummy data
class DummyStdin:
    def readline(self):
        time.sleep(1000)
        return ''

sys.stdin = DummyStdin()

from labelImg.labelImg import main
main()
