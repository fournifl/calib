import cv2
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


def showcv2(img):
    """
    Displays image using openCV's HighGUI

    Parameters
    ----------
    img : Image to display
    """
    cv2.namedWindow("Image", flags=0)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
