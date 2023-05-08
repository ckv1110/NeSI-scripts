# Import modules
from pathml.core import HESlide,\
    SlideData, \
    SlideType, \
    types, \
    h5managers
from pathml.preprocessing import \
    Pipeline, \
    StainNormalizationHE, \
    TissueDetectionHE, \
    LabelArtifactTileHE, \
    LabelWhiteSpaceHE, \
    SegmentMIF, \
    QuantifyMIF, \
    NucleusDetectionHEWs, \
    NucleusDetectionHEWsTest

import numpy
import tqdm
import os
import dask
import dask_mpi as dm
from dask.distributed import Client, LocalCluster
import distributed
# dask.config.set({"distributed.comm.timeouts.tcp": "180s"})


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Load raw .svs file and tile
def load_file(input):
    # fluor_type = SlideType(stain="Fluor", rgb=True)
    # types.HE = fluor_type
    wsi = SlideData(input, name = "test", slide_type = types.HE)
    return wsi

# Define the root pipeline => 1.) Normalize stain, 2.) Detect Tissue 3a.) QC - Look for artifacts 3b.) QC - Look for background white spaces
def root_pipeline():
    pipeline = Pipeline([
        # LabelArtifactTileHE("Artifact"),
        # LabelWhiteSpaceHE("Background", proportion_threshold = 0.3),
        StainNormalizationHE("normalize", "macenko", optical_density_threshold=0.15)
        # NucleusDetectionHEWs("nuclei_mask", "macenko", optical_density_threshold=0.15, superpixel_region_size = 5, n_iter = 30, min_distance=10)
        # NucleusDetectionHEWsTest("nuclei_contour", "macenko", optical_density_threshold=0.01, superpixel_region_size = 7, n_iter = 100, min_distance=10)
    ])
    return pipeline

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # dm.initialize(local_directory='/var/inputdata/')
    # cluster = LocalCluster()
    # client = Client()
    name = 'Raw'
    # name = 'TCGA-14-0789-01Z-00-DX6.dcee0120-1d46-4ab2-a6a8-1906b2c7f1c3'
    size = 512
    save_location = os.path.join(f"/var/outputdata/{name}_{size}.h5path")
    test = load_file(f"/var/inputdata/{name}.tiff")
    # test.run(root_pipeline(), distributed = True, client = client, tile_size = size)
    test.run(root_pipeline(), distributed = False, tile_size = size)
    test.write(save_location)
    print(f"Total number of tiles extracted: {len(test.tiles)}")



