import cv2
import numpy as np
import matplotlib
from src.calib.core.inout import mkdir_p
import src.calib.core.signal as sig

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


def read(fname, gray=None):
    img = cv2.imread(fname)
    if img is None:
        raise OSError(
            "Can't read image file at %s. File might not exist, "
            "or check acces rights" % fname
        )
    if gray is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img


def showmpl(img, ax=None, title=None, **kwargs):
    """
    Display image using Matplotlib's imshow.

    Deal with RGB(matplotlib)/BGR(openCV) problem for color images.

    Parameters
    ----------
    img : Image to display
    """
    if ax is None:
        fig = plt.figure(title)
        ax = fig.add_subplot(111)
    shape = img.shape
    if len(shape) == 3:
        ploted = ax.imshow(img[:, :, ::-1], **kwargs)
    else:
        ploted = ax.imshow(img, **kwargs)
    return ploted


def np2img(array):
    has_alpha = False
    img = array.copy()
    nx, ny, nc = np.squeeze(img).shape
    if nc == 4:
        has_alpha = True
        alpha = img[:, :, -1]
        img = img[:, :, :3]
    img = sig.normalize(img)
    img = img.astype(np.uint8)
    if has_alpha:
        alpha = sig.normalize(alpha)
        alpha = alpha.astype(np.uint8)
        img = np.dstack((img, alpha))
    return img


def save(fname, img):
    import os

    if np.nanmax(img) > 254 or img.dtype != np.uint8:
        img = np2img(img)
    dirname = os.path.dirname(fname)
    if not os.path.isdir(dirname):
        mkdir_p(dirname)
    retval = cv2.imwrite(fname, img)
    if retval is None:
        raise OSError("Can't write image to %s, check access rights" % fname)
    return retval


def show_distortion_field(
    camera_matrix, dist_coeffs, img=None, imshape=None, display=False
):
    """
    From the original image and the intrinsic parameters, return and plot the
    undistorted image

    Parameters
    ----------

    camera_matrix: numpy array of size (3,3)
        The camera intrinsic parameters. (focal length and principal point.
    dist_coeffs numpy array of size 5
        Radial and tangential distortion coefficients
    img : image (np array)
    imshape : tuple
        (nx, ny) or (width, height) in pixels

    Returns
    -------
        fig: matplotlib figure instance to be displayed / saved
    """
    fig = plt.figure(figsize=(12, 9), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    plt.title("Optical distortion field")

    # must reverse vertical axis of image and set origin='lower'
    # otherwise the vertical axis of this mpl.axes would be pointing
    # downward, and the subsequent plots (quiver) would be messed up !!
    if img is not None:
        showmpl(img[::-1, :, :], ax=ax, origin="lower")
        (ny, nx) = img.shape[:2]
    elif imshape is not None:
        (nx, ny) = imshape
    else:
        raise IOError("Need one of reference_img or imshape")

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        np.identity(3),
        camera_matrix,
        (nx, ny),
        cv2.CV_32FC1,
    )

    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    dx = xx - map1
    dy = yy - map2

    res = nx // 40

    contours = ax.contourf(np.sqrt(dx**2 + dy**2), 25, alpha=0.35)
    ax.quiver(x[::res], y[::res], dx[::res, ::res], dy[::res, ::res], color="r")
    cb = plt.colorbar(contours, fraction=0.1, shrink=0.7)
    cb.set_label("pix")
    plt.tight_layout()
    if display:
        plt.show()
    plt.close(fig)
    return fig


def crop(img):
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Créer un masque des pixels non-blancs (ici on considère "blanc" comme proche de 255)
    mask = gray < 254

    # Trouver les coordonnées non-blanches
    coords = np.argwhere(mask)

    # Récupérer les limites (ymin, ymax, xmin, xmax)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # +1 car slicing exclut la borne

    # Recadrer l'image
    cropped = img[y0:y1, x0:x1]

    return cropped
