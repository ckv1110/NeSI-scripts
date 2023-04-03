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
from pathlib import Path
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
        StainNormalizationHE("normalize", "macenko", optical_density_threshold=0.01),
        NucleusDetectionHEWs("nuclei_mask", "macenko", optical_density_threshold=0.01, superpixel_region_size = 5, n_iter = 30, min_distance=10),
        # NucleusDetectionHEWsTest("nuclei_contour", "macenko", optical_density_threshold=0.01, superpixel_region_size = 5, n_iter = 30, min_distance=9)
    ])
    return pipeline

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # loading number of workers and memory limits from the slurm environment
    n_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    total_mem_bytes = int(os.environ["SLURM_MEM_PER_NODE"]) * 1024 * 1024  # SLURM_MEM_PER_NODE is in MB
    mem_limit_bytes = total_mem_bytes // n_workers
    file = os.environ["NAME"]
    name = Path(file).stem
    
    print(f"n_workers: {n_workers}")
    print(f"total memory: {total_mem_bytes / 1024 / 1024} MB")
    print(f"mem limit per worker: {mem_limit_bytes / 1024 / 1024:.0f} MB")
    print(f"file: {name}\n")
    
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True, memory_limit=mem_limit_bytes)
    client = Client(cluster)
    size = 512
    # Feel free to modify the save location to whereever convenient, the current save location is not where I intend to save for the current file
    save_location = os.path.join(f"/var/inputdata/py-data/nuc_wsi_h5/{file}_{size}.h5path")
    test = load_file(f"/var/inputdata/py-data/eval_export_16-01-23/{file}/final_raw/{file}.tif")
    test.run(root_pipeline(), distributed = True, client = client, tile_size = size)
    # test.run(root_pipeline(), distributed = False, tile_size = size)
    print(f"Total number of tiles extracted: {len(test.tiles)}")
    test.write(save_location)




