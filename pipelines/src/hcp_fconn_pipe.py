#!/usr/bin/env python

import utils
import os
from nipype.pipeline.engine import Node, MapNode, Workflow
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.fsl.utils import Smooth, ImageMeants
from nipype.interfaces.fsl.maths import ErodeImage, TemporalFilter
from nipype.interfaces.fsl.preprocess import FAST, ApplyXfm
from nipype.interfaces.fsl.model import GLM
from nipype.interfaces.utility import Function
from nipype.algorithms.misc import Gunzip
from blink_interface import FunctionalConnectivity, AtlasMerger

###
# options
###
options = dict(
    workflow_plugin="Linear",
    sub_args="-q long.q -l h_vmem=15G,virtual_free=10G",
    number_of_processors=2,
    save_graph=False
)

###
# setup basedir
###
basedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.join(basedir, "..", "workspace")
basedir = os.path.realpath(basedir)

###
# scan subjects to process
###
subjects = utils.get_subjects()

###
# scan subjects data
###
subjects_data_fname = os.path.join(basedir, "subjects", "subjects_data.csv")
subjects_data = utils.parse_subject_data(subjects_data_fname)

###
# setup workflow
###
workflow = Workflow(name="hcp_fconn")
workflow.base_dir = os.path.join(basedir, "cache")

infosource = Node(IdentityInterface(fields=["subject_id"]),
                  name="infosource")
infosource.iterables = ("subject_id", subjects)

templates = dict(
    func="{subject_id}/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz",
    struct="{subject_id}/MNINonLinear/T1w_restore_brain.nii.gz",
    brain_mask="{subject_id}/MNINonLinear/brainmask_fs.nii.gz",
    aparc_aseg="{subject_id}/MNINonLinear/aparc+aseg.nii.gz"
)
datasource = Node(SelectFiles(templates), name="datasource")
datasource.inputs.base_directory = os.path.join(basedir, "subjects")

segment = Node(FAST(), name="segment")
segment.inputs.no_pve = True
segment.inputs.segments = True

def del_gm_file(tissue_class_files):
    del tissue_class_files[1]
    return tissue_class_files

erode = MapNode(ErodeImage(), name="erode", iterfield=["in_file"])

sampleseg = MapNode(ApplyXfm(apply_xfm=True), name="sampleseg", iterfield=["in_file"])
sampleseg.inputs.in_matrix_file = os.path.join(basedir, "data", "ident.mat")
sampleseg.inputs.interp = "nearestneighbour"

meants = MapNode(ImageMeants(), name="meants", iterfield=["mask"])

def concat(in_files):
    import os
    from itertools import izip_longest

    assert len(in_files) == 2

    concatted_meants = []
    with open(in_files[0]) as csf_meants_file, open(in_files[1]) as wm_meants_file:
        for line in izip_longest(csf_meants_file, wm_meants_file):
            concatted_meants.append(line[0].strip() + " " + line[1].strip())

    out_fname = os.path.join(os.getcwd(), "csf_wm_meants.txt")
    with open(out_fname, "w") as out_file:
        for line in concatted_meants:
            out_file.write(line + "\n")

    return out_fname

concat = Node(Function(input_names=["in_files"],
                       output_names="out_file",
                       function=concat),
              name="concat")

samplemask = Node(ApplyXfm(apply_xfm=True), name="samplemask")
samplemask.inputs.in_matrix_file = os.path.join(basedir, "data", "ident.mat")
samplemask.inputs.interp = "nearestneighbour"

glm = Node(GLM(), name="glm")
glm.inputs.out_res_name = "func_regressed.nii.gz"

tfilter = Node(TemporalFilter(), name="tfilter")
tfilter.inputs.highpass_sigma = 1389

smooth = Node(Smooth(), name='smoothing')
smooth.inputs.fwhm = 5.0

unzip = Node(Gunzip(), name="unzip")

merger = Node(AtlasMerger(), name="merger")
merger.inputs.atlas2 = os.path.join(basedir, "data", "cerebellum-MNIflirt.nii.gz")
merger.inputs.regions1 = os.path.join(basedir, "data", "aparc+aseg_regions_without_coords.txt")
merger.inputs.regions2 = os.path.join(basedir, "data", "cerebellum-MNIflirt_regions_without_coords.nii.txt")

sampleatlas = Node(ApplyXfm(apply_xfm=True), name="sampleatlas")
sampleatlas.inputs.in_matrix_file = os.path.join(basedir, "data", "ident.mat")
sampleatlas.inputs.interp = "nearestneighbour"

conn = Node(FunctionalConnectivity(), name="conn")

def rewrite_matrix_tojson(in_matrix):
    from nipype.utils.filemanip import split_filename
    import os
    import json

    out_matrix = list()
    with open(in_matrix) as f:
        for row in f:
            row = row.strip()
            if row:
                row = row.split(' ')
                out_matrix.append(row)

    fname = split_filename(in_matrix)[1] + ".json"
    fname = os.path.join(os.getcwd(), fname)
    with open(fname, "w") as f:
        json.dump(out_matrix, f)

    return fname

def rewrite_regions_tojson(in_regions):
    from nipype.utils.filemanip import split_filename
    import os
    import json

    out_regions = list()
    with open(in_regions) as f:
        for row in f:
            row = row.strip()
            if row:
                row = row.split("\t")
                out_regions.append(row)

    fname = split_filename(in_regions)[1] + ".json"
    fname = os.path.join(os.getcwd(), fname)
    with open(fname, "w") as f:
        json.dump(out_regions, f)

    return fname

def create_network_properties(subject_id, subjects_data):
    import os
    import json

    preproc = ("CSF, WM regressed; highpass filtering (sigma 1389); "
               "smoothing (FWHM 5mm); pearson correlation; Fisher Z tranformation; "
               "see https://github.com/medihack/blink for full Nipype pipeline")

    notes = ("Data were provided by the Human Connectome Project, WU-Minn Consortium "
             "(Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) "
             "funded by the 16 NIH Institutes and Centers that support the NIH Blueprint "
             "for Neuroscience Research; and by the McDonnell Center for Systems "
             "Neuroscience at Washington University.")

    subj_data = subjects_data[subject_id]

    props = dict(
        title=subject_id + " (rfMRI_REST1_LR)",
        project="HCP Connectivity Evaluation",
        modality="fmri",
        atlas="Desikan atlas from FSL and Cerebellum from SUIT (aparc_aseg_crbl)",
        subject_type="single",
        gender=subj_data["gender"],
        age=subj_data["age"],
        preprocessing=preproc,
        notes=notes
    )

    props_fname = os.path.join(os.getcwd(), "network_properties.json")
    with open(props_fname, "w") as f:
        json.dump(props, f)

    return props_fname

networkprops = Node(Function(input_names=["subject_id", "subjects_data"],
                             output_names=["network_properties"],
                             function=create_network_properties),
                    name="network_props")
networkprops.inputs.subjects_data = subjects_data

datasink = Node(DataSink(), name="sinker")
datasink.inputs.base_directory = basedir + "/outputs"
datasink.inputs.parameterization = False
datasink.inputs.substitutions = [
    ("network_properties", "network"),
    ("normalized_matrix", "matrix"),
]

# debug node that is normally not used
def debug(input):
    print type(input)
    print input
    return None

debug = Node(Function(input_names=["input"],
                      output_names=[],
                      function=debug),
             name="debug")

workflow.connect([(infosource, datasource, [("subject_id", "subject_id")]),

                  # regress out csf and wm
                  (datasource, segment, [("struct", "in_files")]),
                  (segment, erode, [(("tissue_class_files", del_gm_file), "in_file")]),
                  (erode, sampleseg, [("out_file", "in_file")]),
                  (datasource, sampleseg, [("func", "reference")]),
                  (sampleseg, meants, [("out_file", "mask")]),
                  (datasource, meants, [("func", "in_file")]),
                  (meants, concat, [("out_file", "in_files")]),
                  (datasource, samplemask, [("brain_mask", "in_file")]),
                  (datasource, samplemask, [("func", "reference")]),
                  (samplemask, glm, [("out_file", "mask")]),
                  (concat, glm, [("out_file", "design")]),
                  (datasource, glm, [("func", "in_file")]),

                  # bandpass filter
                  (glm, tfilter, [("out_res", "in_file")]),

                  # smooth
                  (tfilter, smooth, [("out_file", "in_file")]),

                  # unzip nifti file
                  (smooth, unzip, [("smoothed_file", "in_file")]),

                  # merge atlas aparc+aseg and cerebellum
                  (datasource, merger, [("aparc_aseg", "atlas1")]),

                  # resample atlas to voxel size of functional data
                  (merger, sampleatlas, [("merged_atlas", "in_file")]),
                  (datasource, sampleatlas, [("func", "reference")]),

                  # input resampled atlas into calculation of connectivity data
                  (sampleatlas, conn, [("out_file", "atlas")]),

                  # calculate connectivity
                  (unzip, conn, [("out_file", "fmri")]),
                  (merger, conn, [("merged_regions", "defined_regions")]),

                  # write network properties
                  (infosource, networkprops, [("subject_id", "subject_id")]),

                  # save results
                  (infosource, datasink, [("subject_id", "container")]),
                  (conn, datasink, [(("normalized_matrix", rewrite_matrix_tojson), "rfMRI_Rest1_LR.@m")]),
                  (conn, datasink, [(("mapped_regions", rewrite_regions_tojson), "rfMRI_Rest1_LR.@r")]),
                  (networkprops, datasink, [("network_properties", "rfMRI_Rest1_LR.@p")]),
                  ])

if options["save_graph"]:
    workflow.write_graph(graph2use="flat")

workflow.run(
    plugin=options["workflow_plugin"],
    plugin_args={
        "n_procs": options["number_of_processors"],
        "qsub_args": options["sub_args"],
        "bsub_args": options["sub_args"]
    }
)

print "Workflow finished."
