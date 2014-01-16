#!/usr/bin/env python

import sys
import re
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
from blink_interface import FunctionalConnectivity, RegionsMapper

if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
    print "Please provide a file with subject ids (one per line)."
    print "May be created with './manage_subjects -l path_to_subjects_folder'"
    sys.exit(2)

subjects = []

with open(sys.argv[1]) as subjects_file:
    for line in subjects_file:
        line = line.strip()
        line = re.sub(r"#.*", "", line) # remove comments
        if not line:
            continue
        elif re.match(r"^\d", line):
            subjects.append(line)
        else:
            print "Invalid subject id: " + line
            sys.exit(2)

basedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.join(basedir, "..", "workspace")
basedir = os.path.realpath(basedir)

metaflow = Workflow(name="hcp_fconn_preproc")
metaflow.base_dir = os.path.join(basedir, "cache")

infosource = Node(IdentityInterface(fields=["subject_id"]),
                  name="infosource")
infosource.iterables = ("subject_id", subjects)

templates = dict(
    func="{subject_id}/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz",
    struct="{subject_id}/MNINonLinear/T1w_restore_brain.nii.gz",
    brain_mask="{subject_id}/MNINonLinear/brainmask_fs.nii.gz"
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

conn = Node(FunctionalConnectivity(), name="conn")
conn.inputs.atlas = os.path.join(basedir, "data", "aal2mni_2mm.nii.gz")

mapper = Node(RegionsMapper(), name="mapper")
mapper.inputs.definitions = os.path.join(basedir, "data", "aal_regions_with_coords.txt")
mapper.inputs.atlas = os.path.join(basedir, "data", "aal2mni_2mm.nii.gz")

def export_conn(normalized_matrix, regions):
    import os
    import json

    assert len(normalized_matrix.shape) == 2
    nm_fname = os.path.join(os.getcwd(), "normalized_matrix.json")
    with open(nm_fname, "w") as f:
        json.dump(normalized_matrix.tolist(), f)

    assert len(regions.shape) == 1
    r_fname = os.path.join(os.getcwd(), "regions.json")
    with open(r_fname, "w") as f:
        json.dump(regions.tolist(), f)

    return (nm_fname, r_fname)

exportconn = Node(Function(input_names=["normalized_matrix", "regions"],
                       output_names=["normalized_matrix", "regions"],
                       function=export_conn),
              name="export_conn")

def create_network_properties(subject_id):
    import os
    import json

    preproc = "CSF, WM regressed; highpass filtering (sigma 1389); " +\
            "smoothing (FWHM 5mm); pearson correlation; Fisher Z tranformation; " +\
            "see https://github.com/medihack/blink for full Nipype pipeline"

    props = dict(
        title=subject_id + " (rfMRI_REST1_LR)",
        project="HCP Connectivity Evaluation",
        modality="fmri",
        atlas="Automated Anatomical Labeling (AAL)",
        subject_type="single",
        preprocessing=preproc
    )

    props_fname = os.path.join(os.getcwd(), "network_properties.json")
    with open(props_fname, "w") as f:
        json.dump(props, f)

    return props_fname

networkprops = Node(Function(input_names=["subject_id"],
                             output_names=["network_properties"],
                             function=create_network_properties),
                    name="network_props")

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

metaflow.connect([(infosource, datasource, [("subject_id", "subject_id")]),

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

                  # calculate connectivity data
                  (unzip, conn, [("out_file", "fmri")]),

                  # map region ids to label, full_name, x, y, z
                  (conn, mapper, [("regions", "regions")]),

                  # jsonify connectivity data
                  (conn, exportconn, [("normalized_matrix", "normalized_matrix")]),
                  (mapper, exportconn, [("mapped_regions", "regions")]),

                  # write network properties
                  (infosource, networkprops, [("subject_id", "subject_id")]),

                  # save results
                  (infosource, datasink, [("subject_id", "container")]),
                  (exportconn, datasink, [("normalized_matrix", "rfMRI_Rest1_LR.@m")]),
                  (exportconn, datasink, [("regions", "rfMRI_Rest1_LR.@r")]),
                  (networkprops, datasink, [("network_properties", "rfMRI_Rest1_LR.@p")]),
                  ])

metaflow.write_graph(graph2use="flat")
metaflow.run(plugin="MultiProc")

print "Finished."
