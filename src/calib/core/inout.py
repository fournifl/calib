import os
import errno
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)


def mkdir_p(path):
    """
    Recursive mkdir, no error if directory already exists

    Parameters
    ----------
    path : str
        path+name to the directory to be created
    """
    try:
        logger.debug("Making directory %s" % path)
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if (exc.errno == errno.EEXIST and os.path.isdir(path)) or not os.path.dirname(
            path
        ):
            pass
        else:
            raise


def write_dict_to_json(dict, filename):
    """

    Parameters
    ----------
    dict : dictionnary with np array values
    filename : string

    Returns
    -------
    None
    """
    # convert dict value to list
    for key in dict.keys():
        try:
            dict[key] = dict[key].tolist()
        except AttributeError:
            pass
    if not os.path.isdir(os.path.dirname(filename)):
        mkdir_p(os.path.dirname(filename))
    with open(filename, "w") as outfile:
        json.dump(dict, outfile, indent=4)


def read_json_to_dict(filename):
    """

    Parameters
    ----------
    filename : string
        name of the file to load

    Returns
    -------
    dict : dictionnary containing numpy arrays
    """
    logger.info("Reading json file %s" % filename)
    try:
        with open(filename, "r") as infile:
            dict = json.load(infile)
            for k in dict.keys():
                if type(dict[k]) is list:
                    dict[k] = np.array(dict[k])
            return dict
    except IOError:
        raise
