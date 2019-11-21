# Calibrate hyperspectral image data

import os
import cv2
import numpy as np
from plantcv.plantcv import params
from plantcv.plantcv import plot_image
from plantcv.plantcv import print_image
from plantcv.plantcv import Spectral_data
from plantcv.plantcv.transform import rescale


def calibrate(raw_data, white_reference, dark_reference):
    """This function allows you calibrate raw hyperspectral image data with white and dark reference data.

    Inputs:
    raw_data        = Raw image 'Spectral_data' class instance
    white_reference = White reference 'Spectral_data' class instance
    dark_reference  = Dark reference 'Spectral_data' class instance

    Returns:
    calibrated      = Calibrated hyperspectral image

    :param raw_data: __main__.Spectral_data
    :param white_reference: __main__.Spectral_data
    :param dark_reference: __main__.Spectral_data
    :return calibrated: __main__.Spectral_data
    """
    # Auto-increment device
    params.device += 1

    # Store debugging mode
    debug = params.debug

    d_reference = dark_reference
    w_reference = white_reference

    # Collect the number of wavelengths present
    num_bands = len(w_reference.wavelength_dict)
    den = w_reference.array_data - d_reference.array_data

    # Calibrate using reflectance = (raw data - dark reference) / (white reference - dark reference)
    output_num = []
    for i in range(0, raw_data.lines):
        ans = raw_data.array_data[i,] - d_reference.array_data
        output_num.append(ans)
    num = np.stack(output_num, axis=2)
    output_calibrated = []
    for i in range(0, raw_data.lines):
        ans1 = raw_data.array_data[i,] / den
        output_calibrated.append(ans1)

    # Reshape into hyperspectral datacube
    scalibrated = np.stack(output_calibrated, axis=2)
    calibrated_array = np.transpose(scalibrated[0], (1, 0, 2))

    # Make pseudo-rgb image for the calibrated image, take 3 wavelengths, first, middle and last available wavelength
    id_red = len(raw_data.wavelength_dict) - 1
    id_green = int(id_red / 2)
    pseudo_rgb = cv2.merge((calibrated_array[:, :, [0]],
                            calibrated_array[:, :, [id_green]],
                            calibrated_array[:, :, [id_red]]))

    # Gamma correct pseudo_rgb image
    pseudo_rgb = pseudo_rgb ** (1 / 2.2)
    # Scale each of the channels up to 255
    pseudo_rgb = cv2.merge((rescale(pseudo_rgb[:, :, 0]),
                            rescale(pseudo_rgb[:, :, 1]),
                            rescale(pseudo_rgb[:, :, 2])))

    # Make a new class instance with the calibrated hyperspectral image
    calibrated = Spectral_data(array_data=calibrated_array, max_wavelength=raw_data.max_wavelength,
                               min_wavelength=raw_data.min_wavelength, d_type=raw_data.d_type,
                               wavelength_dict=raw_data.wavelength_dict, samples=raw_data.samples,
                               lines=raw_data.lines, interleave=raw_data.interleave,
                               wavelength_units=raw_data.wavelength_units, array_type=raw_data.array_type,
                               pseudo_rgb=raw_data.pseudo_rgb, filename=None)

    # Restore debug mode
    params.debug = debug

    if params.debug == "plot":
        # Gamma correct pseudo_rgb image
        plot_image(pseudo_rgb)
    elif params.debug == "print":
        print_image(pseudo_rgb, os.path.join(params.debug_outdir, str(params.device) + "_calibrated_rgb.png"))

    return calibrated
