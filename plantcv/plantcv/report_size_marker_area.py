# Analyzes an object and outputs numeric properties

import cv2
import numpy as np
import os
from plantcv.plantcv import fatal_error
from plantcv.plantcv import find_objects
from plantcv.plantcv import roi_objects
from plantcv.plantcv import object_composition
from plantcv.plantcv._debug import _debug
from plantcv.plantcv import params
from plantcv.plantcv import outputs


def report_size_marker_area(img, roi_contour, roi_hierarchy, marker='define', masked_img=None, label="default"):
    """
    Detects a size marker in a specified region and reports its size and eccentricity

    Inputs:
    img             = An RGB or grayscale image to plot the marker object on
    roi_contour     = A region of interest contour (e.g. output from pcv.roi.rectangle or other methods)
    roi_hierarchy   = A region of interest contour hierarchy (e.g. output from pcv.roi.rectangle or other methods)
    marker          = 'define' or 'detect'. If define it means you set an area, if detect it means you want to
                      detect within an area
    masked_img      = Binary image in which the size marker is white and the area around it is black.
                      If marker == 'detect', masked_img must != None.
    label      = optional label parameter, modifies the variable name of observations recorded

    Returns:
    analysis_images = List of output images

    :param img: numpy.ndarray
    :param roi_contour: list
    :param roi_hierarchy: numpy.ndarray
    :param marker: str
    :param masked_img: numpy.ndarray
    :param label: str
    :return: analysis_images: list
    """
    # Store debug
    debug = params.debug
    params.debug = None

    params.device += 1
    # Make a copy of the reference image
    ref_img = np.copy(img)
    # If the reference image is grayscale convert it to color
    if len(np.shape(ref_img)) == 2:
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)

    # Marker components
    # If the marker type is "defined" then the marker_mask and marker_contours are equal to the input ROI
    # Initialize a binary image
    roi_mask = np.zeros(np.shape(img)[:2], dtype=np.uint8)
    # Draw the filled ROI on the mask
    cv2.drawContours(roi_mask, roi_contour, -1, (255), -1)
    marker_mask = []
    marker_contour = []

    # If the marker type is "detect" then we will use the ROI to isolate marker contours from the input image
    if marker.upper() == 'DETECT':
        # Confirm that a mask of the size marker has been provided
        if masked_img is not None:
            marker_bin = masked_img
            # Identify contours in the masked image
            contours, hierarchy = find_objects(img=ref_img, mask=marker_bin)
            # Filter marker contours using the input ROI
            kept_contours, kept_hierarchy, kept_mask, obj_area = roi_objects(img=ref_img, object_contour=contours,
                                                                             obj_hierarchy=hierarchy,
                                                                             roi_contour=roi_contour,
                                                                             roi_hierarchy=roi_hierarchy,
                                                                             roi_type="partial")
            # If there are more than one contour detected, combine them into one
            # These become the marker contour and mask
            marker_contour, marker_mask = object_composition(img=ref_img, contours=kept_contours,
                                                             hierarchy=kept_hierarchy)
        else:
            # Reset debug mode
            params.debug = debug
            fatal_error('masked_img must be defined in detect mode')
    elif marker.upper() == "DEFINE":
        # Identify contours in the masked image
        contours, hierarchy = find_objects(img=ref_img, mask=roi_mask)
        # If there are more than one contour detected, combine them into one
        # These become the marker contour and mask
        marker_contour, marker_mask = object_composition(img=ref_img, contours=contours, hierarchy=hierarchy)
    else:
        # Reset debug mode
        params.debug = debug
        fatal_error("marker must be either 'define' or 'detect' but {0} was provided.".format(marker))

    # Calculate the moments of the defined marker region
    m = cv2.moments(marker_mask, binaryImage=True)
    # Calculate the marker area
    marker_area = m['m00']

    # Fit a bounding ellipse to the marker
    center, axes, angle = cv2.fitEllipse(marker_contour)
    major_axis = np.argmax(axes)
    minor_axis = 1 - major_axis
    major_axis_length = axes[major_axis]
    minor_axis_length = axes[minor_axis]
    # Calculate the bounding ellipse eccentricity
    eccentricity = np.sqrt(1 - (axes[minor_axis] / axes[major_axis]) ** 2)

    cv2.drawContours(ref_img, marker_contour, -1, (255, 0, 0), 5)
    analysis_image = ref_img

    # Reset debug mode
    params.debug = debug

    _debug(visual=ref_img,
           filename=os.path.join(params.debug_outdir, str(params.device) + '_marker_shape.png'))

    outputs.add_observation(sample=label, variable='marker_area', trait='marker area',
                            method='plantcv.plantcv.report_size_marker_area', scale='pixels', datatype=int,
                            value=marker_area, label='pixels')
    outputs.add_observation(sample=label, variable='marker_ellipse_major_axis',
                            trait='marker ellipse major axis length',
                            method='plantcv.plantcv.report_size_marker_area', scale='pixels', datatype=int,
                            value=major_axis_length, label='pixels')
    outputs.add_observation(sample=label, variable='marker_ellipse_minor_axis',
                            trait='marker ellipse minor axis length',
                            method='plantcv.plantcv.report_size_marker_area', scale='pixels', datatype=int,
                            value=minor_axis_length, label='pixels')
    outputs.add_observation(sample=label, variable='marker_ellipse_eccentricity', trait='marker ellipse eccentricity',
                            method='plantcv.plantcv.report_size_marker_area', scale='none', datatype=float,
                            value=eccentricity, label='none')

    # Store images
    outputs.images.append(analysis_image)

    return analysis_image
