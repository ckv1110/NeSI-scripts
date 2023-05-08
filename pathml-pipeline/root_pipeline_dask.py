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
        LabelArtifactTileHE("Artifact"),
        LabelWhiteSpaceHE("Background", proportion_threshold = 0.3),
        StainNormalizationHE("normalize", "macenko", optical_density_threshold=0.02),
        NucleusDetectionHEWs("nuclei_mask", "macenko", optical_density_threshold=0.02, superpixel_region_size = 5, n_iter = 30, min_distance=9)
        # NucleusDetectionHEWsTest("nuclei_contour", "macenko", optical_density_threshold=0.02, superpixel_region_size = 7, n_iter = 100, min_distance=10)
    ])
    return pipeline

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # loading number of workers and memory limits from the slurm environment
    n_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    total_mem_bytes = int(os.environ["SLURM_MEM_PER_NODE"]) * 1024 * 1024  # SLURM_MEM_PER_NODE is in MB
    mem_limit_bytes = total_mem_bytes // n_workers
    print(f"n_workers: {n_workers}")
    print(f"total memory: {total_mem_bytes / 1024 / 1024} MB")
    print(f"mem limit per worker: {mem_limit_bytes / 1024 / 1024:.0f} MB")

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True, memory_limit=mem_limit_bytes)
    client = Client(cluster)
    size = 512
    save_location = os.path.join("/var/inputdata/py-data/whole-h5/TCGA-14-0789-preprocessed_{size}.h5path")
    test = load_file("/var/inputdata/WSI-raw/TCGA-02-0003-01Z-00-DX1.6171b175-0972-4e84-9997-2f1ce75f4407.svs")
    test.run(root_pipeline(), distributed = True, client = client, tile_size = size)
    # test.run(root_pipeline(), distributed = False, tile_size = size)
    test.write(save_location)
    print(f"Total number of tiles extracted: {len(test.tiles)}")



