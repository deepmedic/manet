# encoding: utf-8
import SimpleITK as sitk
import logging

import os
logger = logging.getLogger(__name__)

_DICOM_MODALITY_TAG = '0008|0060'
_DICOM_VOI_LUT_FUNCTION = '0028|1056'
_DICOM_WINDOW_CENTER_TAG = '0028|1050'
_DICOM_WINDOW_WIDTH_TAG = '0028|1051'
_DICOM_WINDOW_CENTER_WIDTH_EXPLANATION_TAG = '0028|1055'


def apply_window_level(sitk_image, out_range=[0, 255]):
    """Apply window and level to a SimpleITK image.

    Parameters
    ----------
    sitk_image : SimpleITK image instance
    out_range : tuple or list of new range

    Returns
    -------
    SimpleITK image
    """

    center = float(sitk_image.GetMetaData(
        _DICOM_WINDOW_CENTER_TAG).strip())
    width = float(sitk_image.GetMetaData(
        _DICOM_WINDOW_WIDTH_TAG).strip())

    lower_bound = center - (width - 1)/2
    upper_bound = center + (width - 1)/2

    sitk_image = sitk.IntensityWindowing(
        sitk_image, lower_bound, upper_bound,
        out_range[0], out_range[1])
    # Recast after intensity windowing.
    if (out_range[0] >= 0) and (out_range[1] <= 255):
        pass
    else:
        raise NotImplementedError('Only uint8 supported.')

    sitk_image = sitk.Cast(sitk_image, sitk.sitkUInt8)
    return sitk_image


def read_dcm(filename, window_leveling=True, dtype=None, **kwargs):
    """Read single dicom files. Tries to apply VOILutFunction if available.
    Check if the file is a mammogram or not.

    TODO: Rename to read_mammo and rebuild the read_dcm function.
    TODO: Seperate function to only read the dicom header.
    """
    if not os.path.splitext(filename)[1] == '.dcm':
        raise ValueError('{} should have .dcm as an extension'.format(filename))

    # SimpleITK has issues with unicode string names.
    sitk_image = sitk.ReadImage(filename.encode('utf-8'))
    try:
        modality = sitk_image.GetMetaData(_DICOM_MODALITY_TAG)
    except RuntimeError as e:  # The key probably does not exist
        modality = None
        logger.debug('Modality tag {} does not exist: {}'
                     .format(_DICOM_MODALITY_TAG, e))
    try:
        voi_lut_func = sitk_image.GetMetaData(
            _DICOM_VOI_LUT_FUNCTION).strip()
    except RuntimeError:
        voi_lut_func = 'LINEAR'

    if voi_lut_func != 'LINEAR':
        raise NotImplementedError(
            '{}: VOILutFunction {} not implemented.'.format(filename, voi_lut_func))

    # Check if kwargs contains extra dicom tags
    dicom_keys = kwargs.get('dicom_keys', None)
    extra_metadata = {}
    if dicom_keys:
        for k, v in dicom_keys:
            extra_metadata[k] = sitk_image.GetMetaData(v)

    # This needs to be done after reading all tags.
    # The DICOM tags are lost after this operation.
    if window_leveling:
        try:
            sitk_image = apply_window_level(sitk_image)
        except NotImplementedError as e:
            raise NotImplementedError(
                '{}: {}'.format(filename, e))

    metadata = {}
    metadata.update(extra_metadata)
    metadata['filename'] = os.path.abspath(filename)
    metadata['depth'] = sitk_image.GetDepth()
    metadata['modality'] = 'n/a' if not modality else modality
    metadata['spacing'] = sitk_image.GetSpacing()

    data = sitk.GetArrayFromImage(sitk_image)
    if dtype:
        data = data.astype(dtype)

    if modality == 'MG':
        # If modality is MG the image can be a DBT image.
        # If the image is true mammogram, we reshape.
        if metadata['depth'] == 1:
            data = data.reshape(data.shape[1:])
            metadata['spacing'] = metadata['spacing'][:2]
    else:
        raise NotImplementedError(
            '{}: Modality {} not implemented'.format(filename, modality))

    return data, metadata


def read_dcm_series(path, series_id=None):
    """Read dicom series from a folder. If multiple dicom series are availabe in the folder,
    no image is returned. The metadata dictionary then contains the SeriesIDs which can be selected.

    Parameters
    ----------
    path : str
        path to folder containing the series
    series_ids : str
        SeriesID to load

    Returns
    -------
    metadata dictionary and image as ndarray.

    TODO
    ----
    Catch errors such as
    WARNING: In /tmp/SimpleITK-build/ITK/Modules/IO/GDCM/src/itkGDCMSeriesFileNames.cxx, line 109
    GDCMSeriesFileNames (0x4a6e830): No Series were found
    """

    if not os.path.isdir(path):
        raise ValueError('{} is not a directory'.format(path))

    metadata = {}
    reader = sitk.ImageSeriesReader()
    series_ids = list(reader.GetGDCMSeriesIDs(path.encode('utf-8')))
    metadata['series_ids'] = series_ids
    if len(series_ids) > 1 and not series_ids:
        image = None
        return image, metadata

    fns = reader.GetGDCMSeriesFileNames(
        path.encode('utf-8'), series_id or series_ids[0])
    reader.SetFileNames(fns)
    sitk_image = reader.Execute()

    metadata['filenames'] = fns
    metadata['depth'] = sitk_image.GetDepth()
    # metadata['modality'] = 'n/a' if not modality else modality
    metadata['spacing'] = tuple(sitk_image.GetSpacing()[::-1])

    data = sitk.GetArrayFromImage(sitk_image)
    metadata['shape'] = tuple(data.shape)

    return data, metadata
