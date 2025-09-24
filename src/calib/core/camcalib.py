import logging

import cv2
import numpy as np
import os
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from calib.core.img import read, showcv2, showmpl, crop


logger = logging.getLogger(__name__)


def get_control_points_from_img(path, chessboard_size):
    """Analyse images in path and retrieve pixel coordinates of chessboard
    corners.

    All files in path are analysed (so the directory should contain only
    calibration images). All the calibration images MUST have the same
    dimensions (pixel number).

    To find the chessboard corners coordinates, the algorithm needs to know
    the chessboard size (number of square corners in each dimension),
    and the chessboard in the image should not be larger than the specified
    size. For the following chessboard, dimensions would be (8, 2)::

    |X X X X X|
    | X X X X |
    |X X X X X|

    For each image analyzed, if a chessboard pattern is found, a numpy array is
    appended to the returned list `imgpoints`, containing the pixel
    coordinates of the found corners, and the corresponding image filename
    is appended to the returned list `imgfiles`.

    For a good calibration process, take care to have a good coverage of
    the image (this means taking pictures with the calibration chessboard in
    the corners and borders of the image).


    Parameters
    ----------
    path : str
        path to directory containing calibration images

    chessboard_size : tuple
        number of chessboard corners in the two directions (nx, ny)


    Returns
    -------
    imgpoints : list
        a list of numpy arrays. Each element of the list contains the pixel
        coordinates of the chessboard corners of a calibration image
    imgfiles : list
        image filenames for which chessboard corners were found (the
        corresponding corners coordinates are returner in `imgpoints` list)
    imshape : tuple
        the shape of calibration images. as (width, height)
    """

    nx, ny = chessboard_size
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Arrays to store object points and image points from all the images.
    imgpoints = []  # 2d points in image plane.
    imgfiles = []
    imshape = None

    for fname in os.listdir(path):
        logger.info("Searching calibration pattern in image %s" % fname)
        fname = os.path.join(path, fname)
        try:
            img = read(fname)
        except OSError:
            logger.info("%s is not an image file or is unreachable" % fname)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imshape = gray.shape[::-1]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (ny, nx), None)

        # If found, add object points, image points (after refining them)
        if ret is True:
            logger.info("found points in img : {fname}".format(fname=fname))
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            imgfiles.append(fname)
    return imgpoints, imgfiles, imshape


def hlines(xx, yy):
    ny, nx = xx.shape
    lines = []
    for l in range(ny):
        xs = xx[l, :]
        ys = yy[l, :]
        line = np.stack((xs, ys), axis=0).T.astype(np.int32)
        lines.append(line)
    return lines


def vlines(xx, yy):
    ny, nx = xx.shape
    lines = []
    for c in range(nx):
        xs = xx[:, c]
        ys = yy[:, c]
        line = np.stack((xs, ys), axis=0).T.astype(np.int32)
        lines.append(line)
    return lines


def add_grid_and_undistort(mtx, dist, img):
    distorted = img.copy()
    (ny, nx) = distorted.shape[0:2]
    n = 20
    x = np.linspace(0, nx - 1, n, endpoint=True).astype(np.int16)
    y = np.linspace(0, ny - 1, n, endpoint=True).astype(np.int16)

    map1, map2 = cv2.initUndistortRectifyMap(
        mtx, dist, np.identity(3), mtx, (nx, ny), cv2.CV_32FC1
    )
    map1 = np.where(map1 > 0, map1, 1)
    map2 = np.where(map2 > 0, map2, 1)
    map1 = np.where(map1 < nx, map1, nx - 1)
    map2 = np.where(map2 < ny, map2, ny - 1)
    h_curved = hlines(map1[y, :][:, x], map2[y, :][:, x])
    v_curved = vlines(map1[y, :][:, x], map2[y, :][:, x])
    cv2.polylines(distorted, h_curved, False, (255, 255, 255), thickness=2)
    cv2.polylines(distorted, v_curved, False, (255, 255, 255), thickness=2)

    undistorted = cv2.undistort(distorted, mtx, dist)

    return distorted, undistorted


def get_control_points_from_files(path):
    img_points = []
    obj_points = []
    fnames = []
    for f in os.listdir(path):
        fname = os.path.join(path, f)
        img, obj, __a, __b, imshape = read_control_points(fname)
        img_points.append(np.expand_dims(img, axis=1))
        obj_points.append(np.expand_dims(obj, axis=1))
        if not imshape:
            raise IOError("you must set imshape in control points files")
        fnames.append(f)
    return img_points, obj_points, fnames, imshape


def read_control_points(filename):
    """
    Read the file containing the control points, each line corresponding to
    a point, with optional header informations:
    # beach orientation = 150
    # utmzone = 28N
    # imshape = width x height
    lat lon altitude xpix ypix
    (xpix, ypix)=(0,0) is  left corner

    Parameters
    ----------
    filename : string

    Returns
    -------
    Xpix: ndarray
        shape (npts, 2), each line containing [x, y] pixel coordinates
    GCPs: ndarray
        shape (npts, 3), each line containing [x_geo, y_geo, z_geo] coordinates
    utmzone: str
        defining the utm zone name
    beach_orientation: float
        beach orientation in degrees
    imshape : tuple(ints)
        width, height of image
    """
    utmzone = ""
    beach_orientation = 0
    imshape = ()
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                if "utm" in line:
                    utmzone = line.split("=")[1].strip()
                elif "beach" in line:
                    beach_orientation = line.strip().split("=")[1]
                elif "shape" in line:
                    imshape = line.strip().split("=")[1].split("x")
                    imshape = tuple([int(x.strip()) for x in imshape])

    data = np.genfromtxt(filename, dtype=np.float32)
    npts = data.shape[0]
    Xpix = np.array(data[:, 3:])
    GCPs = np.array(data[:, :3])
    # GCPs[:,[0,1]] = GCPs[:,[1,0]]
    logger.info("Number of Control Points {0}".format(data.shape[0]))
    return Xpix, GCPs, utmzone, float(beach_orientation), imshape


def check_control_points(
    imgpoints, imgfiles, chessboard_size, output_dir, user_input=False
):
    """Visual control of control points found in calibration images by
    :func:`get_control_points_from_img` function.

    Each calibration image is displayed with the found control points, and the
    user is asked to validate or discard the image.

    `imgpoints` and `imgfiles` are returned with the invalid (images/control
    points) pairs discarded

    Parameters
    ----------
    imgpoints : list of np.array
        list of control points as returned by :func:`get_control_points_from_img`
    imgfiles : list of str
        list of image filenames as returned by :func:`get_control_points_from_img`
    chessboard_size : tuple of int
        (nx, ny) points in calibration chessboard pattern

    Returns
    -------
    imgpoints : list of np.array
        list of control points (same as input with only valid points)
    imgfiles : list of str
        list of image filenames (same as input with only valid points)
    """
    check_control_pts_dir = output_dir / "check_control_points"
    check_control_pts_dir.mkdir(parents=True, exist_ok=True)
    nx, ny = chessboard_size
    points_keep = []
    img_keep = []
    for img_pts, fname in zip(imgpoints, imgfiles):
        img = read(fname)
        plot = cv2.drawChessboardCorners(img, (ny, nx), img_pts, True)
        plot = crop(plot)
        fig = plt.figure(figsize=(25, 14))
        plt.imshow(plot[:, :, ::-1])
        plt.axis("off")
        if user_input:
            plt.show()
            var = input(" Control points OK in previous image ? ([y]/n):")
            if var.lower() != "n":
                points_keep.append(img_pts)
                img_keep.append(fname)
        else:
            fig.tight_layout()
            fig.savefig(check_control_pts_dir.joinpath(os.path.basename(fname)))
            points_keep.append(img_pts)
            img_keep.append(fname)
            plt.close("all")

    return points_keep, img_keep


def make_object_points(imgpoints, chessboard_size):
    """
    Create list of object points coordinates from list of img points.

    Parameters
    ----------
    imgpoints : list of np.array
        list of control points
    chessboard_size : tuple of int
        (nx, ny) points in calibration chessboard pattern

    Returns
    -------
    objpoints : list of np.array
        list of objects points coordinates
    """
    nx, ny = chessboard_size
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:ny, 0:nx].T.reshape(-1, 2)
    objp = np.ascontiguousarray(objp.reshape(nx * ny, 1, 3))
    objpoints = [objp] * len(imgpoints)
    return objpoints


def projection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """Compute projection error."""
    mean_error = 0
    n_images = len(objpoints)

    for i in range(n_images):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        error /= len(imgpoints2)
        mean_error += error
    return mean_error / n_images


def undistort(fname, camera_mtx, dist_coeffs, out_path=None):
    """Correct input image `fname` from its optical distortions and save
    result image with the same name and `_undistorted` suffix.

    Parameters
    ----------
    fname
    camera_mtx
    dist_coeffs
    out_path

    Returns
    -------

    """
    img = read(fname)
    # undistort
    dst = cv2.undistort(img, camera_mtx, dist_coeffs, None, camera_mtx)
    if out_path:
        name, ext = os.path.splitext(os.path.join(out_path, os.path.basename(fname)))
    else:
        name, ext = os.path.splitext(fname)

    out_fname = name + "_undistorted.jpg"
    cv2.imwrite(out_fname, dst)
    return dst


def _draw(img, imgpts):
    """Draw a unit cube on img."""
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), 255, 3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img


def plot_axis(imgpoints, objpoints, imgfiles, mtx, dist):
    """Plot axis in images"""
    axis = np.float32(
        [
            [0, 0, 0],
            [0, 3, 0],
            [3, 3, 0],
            [3, 0, 0],
            [0, 0, -3],
            [0, 3, -3],
            [3, 3, -3],
            [3, 0, -3],
        ]
    )

    for imgpt, objpt, fname in zip(imgpoints, objpoints, imgfiles):
        img = read(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objpt, imgpt, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = _draw(img, imgpts)
        showcv2(img)


def plot_img_points_cv(img, img_points):
    img_out = img.copy()
    radius = int(img.shape[0] / 200)
    n = len(img_points)
    colors = []
    cm = plt.get_cmap("gist_rainbow")
    for i in range(n):
        tup = cm(1.0 * i / n)[0:3]  # color will now be an RGBA tuple
        tup2 = tuple([int(j * 254) for j in tup])
        colors.append(tup2)

    for i, points in enumerate(img_points):
        if len(points.shape) > 2:
            points = np.squeeze(points)
        for pt in points:
            cv2.circle(
                img_out,
                center=(int(np.round(pt[0])), int(np.round(pt[1]))),
                radius=radius,
                color=colors[i],
                thickness=2,
            )
    return img_out


def intrinsic_parameters(path, chessboard_size, check_img_points=True):
    """
    Given a directory path and a chessboard_size compute camera matrix and
    distortion coefficients

    Parameters
    ----------
    path : string
        directory path where to find the calibration images
    chessboard_size : tuple of int
        (nx, ny) points in calibration chessboard pattern

    Returns
    -------
    None
    """
    # get control points
    imgpoints, imgfiles, imshape = get_control_points_from_img(path, chessboard_size)

    # compute space covered in the image by calibration points
    points_coverage = plot_img_points_cv(read(imgfiles[0]), imgpoints)

    objpoints = make_object_points(imgpoints, chessboard_size)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, imshape, None, None
    )
    mean_error = projection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    logger.info("Camera matrix : \n{mtx}".format(mtx=mtx))
    logger.info("Optical distorsion coefficients: \n{dist}".format(dist=dist))
    logger.info("Average projection error: {err}".format(err=mean_error))

    imshape = read(imgfiles[0]).shape
    height = imshape[0]
    width = imshape[1]
    intrinsec_dict = {
        "camera_matrix": mtx,
        "dist_coeffs": dist,
        "width": width,
        "height": height,
        "error": mean_error,
    }
    return intrinsec_dict, points_coverage, imgpoints, imgfiles


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def coord_sys_on_calibs(mtx, dist, rvecs, tvecs, imgpoints, files=None, imshape=None):
    ploted = []
    axis_scale = 3
    axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
    axis *= axis_scale

    for i, pts in enumerate(imgpoints):
        try:
            img_f = files[i]
            img = read(img_f)
        except (IndexError, OSError):
            img = np.ones(imshape[::-1] + (3,), dtype=np.uint8) + 50
        img = plot_img_points_cv(img, pts)
        axis_proj, jac = cv2.projectPoints(axis, rvecs[i], tvecs[i], mtx, dist)
        dst = draw(img, pts, axis_proj)
        ploted.append(dst)
    return ploted


def plot_calibration_parameters_contribution(camera_matrix, dist_coeffs, img):
    """

    Parameters
    ----------
    dist_coeffs: [k1,k2,p1,p2,k3] radial and tangential distortion coefficients
    img: np.ndarray
        image as a numpy array

    Returns
    -------
    A figure representing distortion field represented on a grid over the
    reference image with the contribution of each parameter independentely

    """

    (ny, nx, _) = img.shape
    # dist_coeffs comes as a list of list?
    [dist_coeffs] = dist_coeffs
    # Dictionary to reference place of coefficient in array dist_coeffs
    # OpenCV definition is [k1, k2, p1, p2, k3 ]
    dist_coeffs_dict = {"k1": 0, "k2": 1, "p1": 2, "p2": 3, "k3": 4}

    fig = plt.figure(figsize=(20, 15), dpi=80)
    count = 1
    distorted, undistorted = add_grid_and_undistort(camera_matrix, dist_coeffs, img)
    ax = fig.add_subplot(2, 3, count)
    ax.set_title("original distorted grid")
    showmpl(distorted, ax=ax)

    for k in ["k1", "k2", "p1", "p2", "k3"]:
        count += 1
        dist_coeff_reduced = np.zeros(5)
        # All parameters are set to zero but one
        dist_coeff_reduced[dist_coeffs_dict[k]] = dist_coeffs[[dist_coeffs_dict[k]]]
        distorted, undistorted = add_grid_and_undistort(
            camera_matrix, dist_coeff_reduced, img
        )
        ax = fig.add_subplot(2, 3, count)
        ax.set_title(k)
        showmpl(distorted, ax=ax)
    plt.tight_layout()
    plt.close(fig)
    return fig
