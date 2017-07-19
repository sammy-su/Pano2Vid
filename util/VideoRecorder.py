#!/usr/bin/env python
# encoding: utf-8

import os
import cv2
import numpy as np

from scipy.interpolate import RegularGridInterpolator as interp2d

class ImageRecorder(object):

    """
    Record normal view video given 360 degree video.
    """

    def __init__(self, sphereW, sphereH, view_angle=65.5, imgW=640):
        self.sphereW = sphereW
        self.sphereH = sphereH

        self._imgW = imgW
        w, h = 4, 3

        self._imgH = h * self._imgW / w
        self._Y = np.arange(self._imgH) + (self._imgW - self._imgH)/2

        TX, TY = self._meshgrid()
        R, ANGy = self._compute_radius(view_angle, TY)

        self._R = R
        self._ANGy = ANGy
        self._Z = TX

    def _meshgrid(self):
        """Construct mesh point
        :returns: TX, TY

        """
        TX, TY = np.meshgrid(range(self._imgW), range(self._imgW))
        TX = TX[self._Y,:]
        TY = TY[self._Y,:]

        TX = TX.astype(np.float64) - 0.5
        TX -= self._imgW/2

        TY = TY.astype(np.float64) - 0.5
        TY -= self._imgW/2
        return TX, TY

    def _compute_radius(self, view_angle, TY):
        _view_angle = np.pi * view_angle / 180.
        r = self._imgW/2 / np.tan(_view_angle/2)
        R = np.sqrt(np.power(TY, 2) + r**2)
        ANGy = np.arctan(-TY/r)
        return R, ANGy

    def catch(self, x, y, image):
        Px, Py = self._sample_points(x, y)
        warped_image = self._warp_image(Px, Py, image)
        return warped_image

    def _sample_points(self, x, y):
        angle_x, angle_y = self._direct_camera(x, y)
        Px = (angle_x + np.pi) / (2*np.pi) * self.sphereW + 0.5
        Py = (np.pi/2 - angle_y) / np.pi * self.sphereH + 0.5
        INDx = Px < 1
        Px[INDx] += self.sphereW
        return Px, Py

    def _direct_camera(self, rotate_x, rotate_y):
        angle_y = self._ANGy + rotate_y
        X = np.sin(angle_y) * self._R
        Y = - np.cos(angle_y) * self._R
        Z = self._Z

        INDn = np.abs(angle_y) > np.pi/2

        angle_x = np.arctan(Z / -Y)
        RZY = np.linalg.norm(np.stack((Y, Z), axis=0), axis=0)
        angle_y = np.arctan(X / RZY)

        angle_x[INDn] += np.pi
        angle_x += rotate_x

        INDy = angle_y < -np.pi/2
        angle_y[INDy] = -np.pi -angle_y[INDy]
        angle_x[INDy] = angle_x[INDy] + np.pi

        INDx = angle_x <= -np.pi
        angle_x[INDx] += 2*np.pi
        INDx = angle_x > np.pi
        angle_x[INDx] -= 2*np.pi
        return angle_x, angle_y

    def _warp_image(self, Px, Py, frame):
        minX = max(0, int(np.floor(Px.min())))
        minY = max(0, int(np.floor(Py.min())))

        maxX = min(int(self.sphereW), int(np.ceil(Px.max())))
        maxY = min(int(self.sphereH), int(np.ceil(Py.max())))

        im = frame[minY:maxY, minX:maxX, :]
        Px -= minX
        Py -= minY
        warped_images = []

        y_grid = np.arange(im.shape[0])
        x_grid = np.arange(im.shape[1])
        samples = np.vstack([Py.ravel(), Px.ravel()]).transpose()
        for c in xrange(3):
            full_image = interp2d((y_grid, x_grid), im[:,:,c],
                                   bounds_error=False,
                                   method='linear',
                                   fill_value=None)
            warped_image = full_image(samples).reshape(Px.shape)
            warped_images.append(warped_image)
        warped_image = np.stack(warped_images, axis=2)
        return warped_image

class VideoRecorder(ImageRecorder):
    def __init__(self, video_in, video_out, x0=0., y0=0., view_angle=65.5):
        video = cv2.VideoCapture(video_in)
        sphereW = video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        sphereH = video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        self.fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        super(VideoRecorder, self).__init__(sphereW, sphereH, view_angle=view_angle)
        self.video = video
        self._current = 0

        self._x = x0
        self._y = y0

        output_path = '{}.avi'.format(video_out)
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        self.output = cv2.VideoWriter(output_path, fourcc, self.fps,
                                      (self._imgW, self._imgH))

    def forward(self, diff_x=0, diff_y=0, distance=None):
        if distance is None:
            distance = int(np.floor(self.fps * 5))

        finished = False
        x_step = float(diff_x) / distance
        y_step = float(diff_y) / distance
        for step in xrange(distance):
            self._displace_x(x_step)
            self._displace_y(y_step)
            warped_image = self._capture_next()
            if warped_image is None:
                finished = True
                break
            self.output.write(warped_image)
        return finished

    def _displace_x(self, diff):
        self._x += diff
        if self._x >= np.pi:
            self._x -= 2*np.pi
        elif self._x < -np.pi:
            self._x += 2*np.pi

    def _displace_y(self, diff):
        self._y += diff

    def _capture_next(self):
        ret, frame = self.video.read()
        if not ret:
            warped_image = None
        else:
            self._current += 1
            warped_image = self.catch(self._x, self._y, frame)
            warped_image[warped_image>255.] = 255.
            warped_image[warped_image<0.] = 0.
            warped_image = warped_image.astype(np.uint8)
        return warped_image

class ZoomCamera(object):
    def __init__(self, video_in, video_out, distance="5s", imgW=640):
        video = cv2.VideoCapture(video_in)
        self.__sphereW = video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        self.__sphereH = video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        self.__fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        self.__video = video
        self.__imgW = imgW

        unit = distance[-1:]
        try:
            value = int(distance[:-1])
        except:
            raise ValueError("Invalid distance format")
        else:
            if unit == "s":
                self.__distance = int(np.floor(self.__fps * value))
            elif unit == "f":
                self.__distance = value
            else:
                raise ValueError("Invalid distance unit {}".format(distance))

        recorder = ImageRecorder(self.__sphereW,
                                 self.__sphereH,
                                 view_angle=65.5,
                                 imgW=self.__imgW)
        self.__recorder = {65.5: recorder}
        self.__recording = False

        self.__x = 0.
        self.__y = 0.
        self.__z = 65.5

        output_path = '{}.avi'.format(video_out)
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        size = (recorder._imgW, recorder._imgH)
        self.__output = cv2.VideoWriter(output_path, fourcc, self.__fps, size)

    def start_record(self, x, y, z=65.5):
        self.__recording = True
        self.__x = x
        self.__y = y
        self.__z = z

    def end_record(self):
        self.__recording = False

    def forward(self, diff_x=0, diff_y=0, diff_z=0):
        finished = False
        dx = float(diff_x) / self.__distance
        dy = float(diff_y) / self.__distance
        dz = float(diff_z) / self.__distance
        for step in xrange(self.__distance):
            self.__displace(dx, dy, dz)

            warped_image = self.__capture_next()
            if warped_image is None:
                finished = True
                break
            if self.__recording:
                self.__output.write(warped_image)
        return finished

    def __displace(self, dx, dy, dz):
        self.__x += dx
        if self.__x >= np.pi:
            self.__x -= 2*np.pi
        elif self.__x < -np.pi:
            self.__x += 2*np.pi
        self.__y += dy
        if self.__y > np.pi or self.__y < -np.pi:
            raise ValueError("Invalid latitude")
        self.__z += dz
        if self.__z <= 0 or self.__z > 180:
            raise ValueError("Invalid angle of view")

    def __capture_next(self):
        ret, frame = self.__video.read()
        if not ret:
            warped_image = None
        else:
            if self.__recorder.has_key(self.__z):
                recorder = self.__recorder[self.__z]
            else:
                recorder = ImageRecorder(self.__sphereW, self.__sphereH,
                                         view_angle=self.__z,
                                         imgW=self.__imgW)
                self.__recorder[self.__z] = recorder
            warped_image = recorder.catch(self.__x, self.__y, frame)
            warped_image[warped_image>255.] = 255.
            warped_image[warped_image<0.] = 0.
            warped_image = warped_image.astype(np.uint8)
        return warped_image

