import struct
import cv2
from pathlib import Path
import numpy as np

from .imutils import load_image


def load_bin(filename, size, ds=4, dtype='f'):
    """
    read a matrix which is saved in binary file

    Args:
    - size: matrix size, for example: (10000, 2)
    - ds: data type size (bytes)
    """
    with open(filename, 'rb') as f:
        data = np.zeros(size)
        with np.nditer(data, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = struct.unpack(dtype, f.read(ds))[0]

    return data


def load_camera_param(folder):
    """
    load intrinsic and extrinsic from intrinsic0.yml and extrinsic0.yml

    Returns:
    - rvec, tvec, cameraMatrix, distCoeffs
    """
    p = Path(folder)

    intrinsic0 = p / "intrinsic0.yml"
    extrinsic0 = p / "extrinsic0.yml"

    fs = cv2.FileStorage(str(intrinsic0), cv2.FILE_STORAGE_READ)
    cameraMatrix = fs.getNode("camera_matrix").mat()
    distCoeffs = fs.getNode("distortion_coefficients").mat()

    fs = cv2.FileStorage(str(extrinsic0), cv2.FILE_STORAGE_READ)
    rvec = fs.getNode("rvec").mat()
    tvec = fs.getNode("tvec").mat()

    return rvec, tvec, cameraMatrix, distCoeffs


def projectPoints():
    imgPts, _ = cv2.projectPoints(data, rvec, tvec, cameraMatrix, distCoeffs)



def parse_gutter_map(file_path):
    """
    read gutter_map, return gutter pos

    Returns:
    - ndarray N × 2
    """
    gutter_map = load_image(file_path, to_tensor=False)
    return np.argwhere(gutter_map[:, :, 0] > 0)

