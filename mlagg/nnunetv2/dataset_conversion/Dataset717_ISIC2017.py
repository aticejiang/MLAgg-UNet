import multiprocessing
import shutil
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes
import numpy as np
from skimage import transform

height = 256
width  = 256
channels = 3

def load_and_covnert_case(input_image: str, input_seg: str, output_image: str, output_seg: str,
                          min_component_size: int = 50):
    seg = io.imread(input_seg[:-4] + '_segmentation.png')
    seg_resize = transform.resize(seg, (height, width), order=0)
    assert np.unique(seg_resize)[0] == 0 and np.unique(seg_resize)[1] == 255
    seg_resize[seg_resize == 255] = 1
    io.imsave(output_seg, seg_resize, check_contrast=False)

    img = io.imread(input_image)
    img_resize = transform.resize(img, (height, width), order=0)
    io.imsave(output_image, img_resize)
    return 0


if __name__ == "__main__":
    source = 'E:/Datasets/ISIC17/dataset'

    dataset_name = 'Dataset717_ISIC2017'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagesval = join(nnUNet_raw, dataset_name, 'imagesVal')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsval = join(nnUNet_raw, dataset_name, 'labelsVal')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagesval)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsval)
    maybe_mkdir_p(labelsts)

    train_source = join(source, 'ISIC-2017_Training_Data')
    train_source_seg = join(source, 'ISIC-2017_Training_Part1_GroundTruth')
    val_source = join(source, 'ISIC-2017_Validation_Data')
    val_source_seg = join(source, 'ISIC-2017_Validation_Part1_GroundTruth')
    # test_source =
    # test_source_seg =

    ############# test ############
    # valid_ids = subfiles(train_source, join=False, suffix='jpg')
    # for v in valid_ids:
    #     load_and_covnert_case(join(train_source, v), join(train_source_seg, v),
    #                           join(imagestr, v[:-4] + '_0000.png'), join(labelstr, v[:-4] + '.png'), 50)

    with multiprocessing.get_context("spawn").Pool(8) as p:

        # not all training images have a segmentation
        valid_ids = subfiles(train_source, join=False, suffix='jpg')
        num_train = len(valid_ids)
        r = []
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(train_source, v),
                         join(train_source_seg, v),
                         join(imagestr, v[:-4] + '_0000.png'),
                         join(labelstr, v[:-4] + '.png'),
                         50
                     ),)
                )
            )

        # val set
        valid_ids = subfiles(val_source, join=False, suffix='jpg')
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(val_source, v),
                         join(val_source_seg, v),
                         join(imagesval, v[:-4] + '_0000.png'),
                         join(labelsval, v[:-4] + '.png'),
                         50
                     ),)
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'Melanoma': 1},
                          num_train, '.png', dataset_name=dataset_name)