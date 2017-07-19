
import os
import sys
import numpy as np
import cv2
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from util.VideoRecorder import ImageRecorder, ZoomCamera

colors = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "cyan": (255, 255, 0),
}
record_dir = "recordings"
result_dir = "results"

class ImageRenderEngine(ImageRecorder):
    def _meshgrid(self):
        TX, TY = np.meshgrid(range(self._imgW), range(self._imgW))
        TX = TX[self._Y,:]
        TY = TY[self._Y,:]

        # sample boundary points
        TXb = np.hstack([TX[0,:], TX[-1,:], TX[:,0], TX[:,-1]])
        TYb = np.hstack([TY[0,:], TY[-1,:], TY[:,0], TY[:,-1]])
        # sample center
        cx, cy = TX.shape[0]/2, TX.shape[1]/2
        TXc = TX[cx, cy]
        TYc = TY[cx, cy]
        TX = np.append(TXb, TXc)
        TY = np.append(TYb, TYc)

        TX = TX.astype(np.float64) - 0.5
        TX -= self._imgW/2

        TY = TY.astype(np.float64) - 0.5
        TY -= self._imgW/2
        return TX, TY

    def catch(self, x, y, image):
        Px, Py = self._sample_points(x, y)
        image = self._draw_image(Px, Py, image)
        return image

    def _draw_image(self, Px, Py, frame):
        TX = np.floor(Px).astype(np.int64)
        TY = np.floor(Py).astype(np.int64)
        TXb = TX[:-1]
        TYb = TY[:-1]
        cPx = TX[-1]
        cPy = TY[-1]
        frame = self._draw_thick_line(frame, TXb, TYb)
        cv2.circle(frame, (cPx, cPy), radius=8, color=colors["red"], thickness=-1)
        # for interface output only
        self.supY = TY.min()
        self.camera_center = (cPx, cPy)
        return frame

    def _draw_thick_line(self, frame, px, py):
        radius = 1
        for dx in xrange(-radius, radius+1):
            for dy in xrange(-radius, radius+1):
                delta = abs(dx) + abs(dy)
                if delta > radius:
                    continue
                frame = self._draw_line(frame, px+dx, py+dy)
        return frame

    def _draw_line(self, frame, px, py):
        px[px>=frame.shape[1]] = frame.shape[1] - 1
        py[py>=frame.shape[0]] = frame.shape[0] - 1
        frame[py,px,:] = colors["cyan"]
        return frame

class VideoRenderEngine(object):
    def __init__(self, video_in, traj_name=None, rotate=0., write_output=True):
        self.recording = traj_name is not None
        self.write_output = write_output

        # Collect necessary input video information
        video = cv2.VideoCapture(video_in)
        sphereW = video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        sphereH = video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        delay = max(1, int(np.floor(1000./fps))-5)
        delay = max(1, delay/2) # Speed up annotation
        self.n_frames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        self.sphereW = sphereW
        self.sphereH = sphereH
        self.delay = delay
        self.video = video

        # set up display parameters
        self.ext_rows = int(sphereW/4)
        self.l_left = self.ext_rows
        self.l_right = self.l_left + int(sphereW)

        wW = get_screen_width()
        scale = wW / (1.5 * sphereW)
        wH = int(np.ceil(scale * sphereH))
        self.roi_left = int(self.l_left * scale)
        self.roi_right = int(self.l_right * scale)
        self.roi_width = self.roi_right - self.roi_left
        self.wW = wW
        self.wH = wH

        # Construct output writer
        if self.recording:
            output_name = "{}.avi".format(traj_name)
            output_path = os.path.join(record_dir, output_name)
            fourcc = cv2.cv.CV_FOURCC(*'XVID')
            size = (wW, wH)
            #self.output = cv2.VideoWriter(output_path, fourcc, fps, size)
            #self.output_path = output_path

            if write_output:
                self.result_path = os.path.join(result_dir, traj_name)
                self.synthesizer = ZoomCamera(video_in, self.result_path,
                                              distance="1f", imgW=160)
                self.rotate = np.pi * rotate / 180.
                self.synthesizer.start_record(self.rotate, 0.)
                self.x0 = self.rotate
                self.y0 = 0.
                self.z0 = 65.5

        self.rotate_shift = -int(sphereW * rotate / 360.)
        self.renderors = {}
        self.t = 1

        # set up display window
        top = 24
        left = 0
        self.w_left = left
        self.w_right = left + wW

        cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Display", wW, wH)
        cv2.moveWindow("Display", left, top)
        ret, frame = self.next_frame()
        self.display(frame)
        sys.stderr.write("\n")
        for _ in xrange(28):
            sys.stderr.write("=")
        sys.stderr.write("\n\n")
        sys.stderr.write("Press \'s\' to start,\n")
        sys.stderr.write("      \'w\' to pause,\n")
        sys.stderr.write("      \'c\' to quit,\n")
        sys.stderr.write("      \'1\' to zoom-out,\n")
        sys.stderr.write("      \'2\' to zoom-in.\n\n")
        while True:
            key = read_key(0)
            if key == ord('s'):
                break
        self.w_top = get_img_top()
        self.w_bot = self.w_top + wH

    def next_frame(self):
        ret, frame = self.video.read()
        if ret:
            h_axis = 1
            frame = np.roll(frame, self.rotate_shift, h_axis)
        return ret, frame

    def display(self, frame):
        if self.t % 100 == 0:
            width = len(str(self.n_frames))
            template = "Progress: {0:{width}d}/{1}\n"
            progress = template.format(self.t, self.n_frames, width=width)
            sys.stderr.write(progress)

        ext_frames = (frame[:,-self.ext_rows:,:], frame, frame[:,:self.ext_rows])
        ext_frame = np.concatenate(ext_frames, axis=1)
        H = frame.shape[0]
        cv2.line(ext_frame, (self.l_left, 0), (self.l_left, H), color=colors["red"])
        cv2.line(ext_frame, (self.l_right, 0), (self.l_right, H), color=colors["red"])
        cv2.imshow('Display', ext_frame)
        #if self.recording:
        #    self.output.write(ext_frame)

    def forward(self, x, y, z):
        start = get_time()
        ret, frame = self.next_frame()
        if ret:
            if self.renderors.has_key(z):
                renderor = self.renderors[z]
            else:
                renderor = ImageRenderEngine(self.sphereW, self.sphereH, z)
                self.renderors[z] = renderor
            if self.recording:
                frame = renderor.catch(x, y, frame)

                ### construct output on the fly
                if self.write_output:
                    x += self.rotate
                    if x > np.pi:
                        x -= 2*np.pi
                    elif x < -np.pi:
                        x += 2*np.pi

                    dx = x - self.x0
                    dy = y - self.y0
                    dz = z - self.z0
                    self.synthesizer.forward(dx, dy, dz)
                    self.x0 = x
                    self.y0 = y
                    self.z0 = z

            self.t += 1
            self.display(frame)

            end = get_time()
            elapsed = end - start
            if elapsed < self.delay:
                delay = self.delay - elapsed
            else:
                delay = 1
            key = read_key(delay)
            if key == ord('w'):
                while True:
                    key = read_key(0)
                    if key == ord('w'):
                        break
            return key
        else:
            raise EOFError("Video end.")

    def forward_screen_coordinate(self, x, y, z=65.5):
        theta_x, theta_y = self.screen_coordinate_to_angle(x, y)
        key = self.forward(theta_x, theta_y, z)
        return theta_x, theta_y, key

    def screen_coordinate_to_angle(self, x, y):
        x = min(self.w_right, max(self.w_left, x))
        y = min(self.w_bot, max(self.w_top, y)) - self.w_top
        if x > self.roi_right:
            x = x - self.roi_right + self.roi_left
        elif x < self.roi_left:
            x = self.roi_right + x - self.roi_left
        x -= self.roi_left
        theta_x = (2.* x - 1.) / self.roi_width - 1.
        angle_x = theta_x * np.pi

        theta_y = 0.5 - (y - 0.5) / self.wH
        angle_y = theta_y * np.pi
        return angle_x, angle_y

    def __del__(self):
        cv2.destroyWindow("Display")
        if hasattr(self, 'output_path'):
            os.chmod(self.output_path, 0o666)

def get_time():
    millis = int(os.times()[-1]*1000)
    return millis

def get_screen_width():
    cmd = '''/usr/bin/python -c "exec('import gtk\\nprint gtk.Window().get_screen().get_monitor_geometry(0).width')"'''
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()
    return int(output)

def get_img_top():
    cmd = "xwininfo -name Display -frame | grep \'Absolute upper-left Y:\'"
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()
    top = output.split()[-1]
    return int(top)

def read_key(wait):
    key = cv2.waitKey(wait)
    key %= 256
    if key == ord('c'):
        raise KeyboardInterrupt
    return key
