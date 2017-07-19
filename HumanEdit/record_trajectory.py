#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode = True
import random

from VideoRecorder import VideoRecorder

def main():
    userId = raw_input("Please input user ID: ")
    categories = ["hiking", "mountain_climbing", "parade", "soccer"]
    for category in categories:
        recorder = VideoRecorder(category, userId)
        video, idx = recorder.find_next_video()
        try:
            video, idx = recorder.find_next_video()
        except:
            continue
        else:
            msg = "\nRecord video {0} in category {1}\n".format(idx, category)
            sys.stderr.write(msg)

            recorder.record_video()
            break

if __name__ == "__main__":
    main()
