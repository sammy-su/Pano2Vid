
import os
import sys
import numpy as np
import subprocess
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from util.data_io import load_pkl, dump_pkl, find_video_path
from VideoRenderEngine import VideoRenderEngine, read_key

traj_dir = os.path.join(ROOT, "HumanEdit", "trajectories")

class VideoRecorder(object):
    def __init__(self, category, userId):
        self.category = category
        self.userId = userId

        self.n_train = 2
        self.rotation_angles = [0., 180., 0., 180.]
        self.num_trajs = len(self.rotation_angles)
        self.num_output = 2

        self.zooms = [46.4, 65.5, 104.3]

    def find_next_video(self):
        index = load_index()
        videos = index[self.category]
        for idx, (videoId, duration) in enumerate(videos, 1):
            if idx > 10:
                break
            remain = 0
            for trajId in xrange(self.num_trajs):
                record_path = self._construct_traj_path(videoId, trajId)
                if not os.path.exists(record_path):
                    remain = self.num_trajs - trajId
                    break
            if remain > 0:
                self.videoId = videoId
                self.remain = remain
                # idx is 1-based
                self.idx = idx
                return videoId, idx
        raise ValueError("No more video in this category.")

    def _construct_traj_path(self, videoId, trajId):
        traj_name = "{0}-{1}-traj{2}.pkl".format(videoId, self.userId, trajId)
        traj_path = os.path.join(traj_dir, traj_name)
        return traj_path

    def record_video(self):
        video_in = find_video_path(self.videoId)

        # Show the video if the annotator first see this video
        if self.remain == self.num_trajs:
            self._record_video(video_in, recording=False)

        start = self.num_trajs - self.remain
        for trajId in xrange(start, self.num_trajs):
            while True:
                recorded = self._record_video(video_in, trajId)
                if recorded:
                    break

    def _record_video(self, video_in, trajId=0, recording=True):
        assert trajId < self.num_trajs
        if recording:
            rotate = self.rotation_angles[trajId]
        else:
            rotate = 0.
        self.rotate = rotate

        if recording:
            traj_path = self._construct_traj_path(self.videoId, trajId)
            record_name = "{0}-{1}-traj{2}".format(self.videoId, self.userId, trajId)
        else:
            record_name = None

        zoom_level = self.zooms.index(65.5)
        write_output = self.idx <= self.n_train and trajId < self.num_output
        video_engine = VideoRenderEngine(video_in,
                                         traj_name=record_name,
                                         rotate=rotate,
                                         write_output=write_output)
        trajectory = []
        if recording:
            tracker = setup_tracker(video_engine)

        while True:
            if recording:
                x, y = tracker.next()
            else:
                x, y = 0, 0
            try:
                z = self.zooms[zoom_level]
                theta_x, theta_y, key = video_engine.forward_screen_coordinate(x, y, z)
            except EOFError:
                sys.stderr.write("\nVideo end. ")
                if recording:
                    if write_output:
                        sys.stderr.write("Press 's' to play the recording.\n")
                        while True:
                            key = read_key(0)
                            if key == ord('s'):
                                subprocess.call(['cvlc', "--rate", "2", '--play-and-exit', "--fullscreen",
                                                 video_engine.result_path+".avi"])
                                break

                    sys.stderr.write("Press 's' to save, 'c' to quit, 'r' to repeat.\n")
                    while True:
                        key = read_key(0)
                        if key == ord('s'):
                            self._finalize_record(trajectory, traj_path)
                            return True
                        elif key == ord('r'):
                            return False
                else:
                    sys.stderr.write("Press any key to continue.\n")
                    read_key(0)
                    return True
            else:
                traj = (theta_x, theta_y, z)
                trajectory.append(traj)
                if key == ord('1'):
                    zoom_level += 1
                    zoom_level = min(zoom_level, len(self.zooms)-1)
                elif key == ord('2'):
                    zoom_level -= 1
                    zoom_level = max(zoom_level, 0)

    def _finalize_record(self, trajs, path):
        trajectory = []
        rotate_angle = np.pi * self.rotate / 180.
        for i, traj in enumerate(trajs, 1):
            theta_x, theta_y, z = traj
            # rotate horizontal
            theta_x += rotate_angle
            if theta_x > np.pi:
                theta_x -= 2*np.pi
            elif theta_x < -np.pi:
                theta_x += 2*np.pi
            trajectory.append((theta_x, theta_y, z))
        trajectory = np.array(trajectory)
        dump_pkl(path, trajectory)

def setup_tracker(video_engine):
    border_left = video_engine.roi_left
    border_right = video_engine.roi_right

    cx = (video_engine.w_left + video_engine.w_right) / 2
    cy = (video_engine.w_top + video_engine.w_bot) / 2
    center = (cx, cy)
    return mouse_tracker(border_left, border_right, center)

def mouse_tracker(border_left, border_right, center):
    from Xlib import X, display
    d = display.Display()
    root = d.screen().root
    root.warp_pointer(*center)
    d.sync()
    while True:
       data = root.query_pointer()._data
       x = data["root_x"]
       y = data["root_y"]
       if x < border_left:
           x = border_right + x - border_left
           root.warp_pointer(x,y)
           d.sync()
       elif x >= border_right:
           x = border_left + x - border_right
           root.warp_pointer(x,y)
           d.sync()
       yield x, y

def load_index():
    index_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'selected_videos.pkl')
    index = load_pkl(index_path)
    return index

