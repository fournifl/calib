from pathlib import Path

import numpy as np

import calib.core.camcalib as cal
import calib.core.img as cimg
from calib.core.inout import mkdir_p, read_json_to_dict, write_dict_to_json


def show_intrinsic(input_dir, output_dir, img_real=True):
    calib_file = output_dir / "calibration.json"

    tmp = read_json_to_dict(calib_file)
    dist = tmp["dist_coeffs"]
    mtx = tmp["camera_matrix"]
    width = tmp["width"]
    height = tmp["height"]
    imshape = (width, height)
    # destination directory
    dest = Path(output_dir.joinpath("show_intrinsic"))
    dest.mkdir(parents=True, exist_ok=True)

    # img ref to show distorsion field and coefficients
    img_ref_show_dist = np.ones(imshape[::-1] + (3,), dtype=np.uint8) + 180
    if img_real:
        img = sorted(input_dir.glob("*.jpeg"))[0]
        img_ref = cimg.read(img)
        assert img_ref.shape[0:2][::-1] == imshape
    else:
        img_ref = img_ref_show_dist

    aa, bb = cal.add_grid_and_undistort(mtx, dist, img_ref)
    fig = cimg.show_distortion_field(mtx, dist, img=img_ref_show_dist, imshape=imshape)
    fig2 = cal.plot_calibration_parameters_contribution(mtx, dist, img_ref_show_dist)

    mkdir_p(dest)
    cimg.save(dest.joinpath("ref.png"), aa)
    cimg.save(dest.joinpath("ref_undistort.png"), bb)
    fig.savefig(dest.joinpath("distortion.png"))
    fig2.savefig(dest.joinpath("distortion_coefficients.png"))


def main(input_directory, output_directory, chessboard_size):
    # compute intrinsic parameters
    intrinsec_dict, cali_pts_coverage, imgpoints, imgfiles = cal.intrinsic_parameters(
        input_directory, chessboard_size
    )

    # check control points
    cal.check_control_points(imgpoints, imgfiles, chessboard_size, output_directory)

    # plot space covered in the image by calibration points
    cimg.save(output_directory.joinpath("calibration_points.png"), cali_pts_coverage)

    # save calibration results to json
    write_dict_to_json(intrinsec_dict, output_directory.joinpath("calibration.json"))

    # show intrinsic
    show_intrinsic(input_directory, output_directory)

    return
