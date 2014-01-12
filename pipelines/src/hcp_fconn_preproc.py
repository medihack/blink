#!/usr/bin/env python

import os
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.fsl.utils import Smooth, ImageMeants
from nipype.interfaces.fsl.maths import TemporalFilter
from nipype.interfaces.fsl.preprocess import FAST
from nipype.interfaces.utility import Function
from nipype.interfaces.fsl import Merge
from nipype.algorithms.misc import Gunzip
from blink_interface import FunctionalConnectivity

subjects = ["100408"]

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
    struct="{subject_id}/MNINonLinear/T1w_restore_brain.nii.gz"
)
datasource = Node(SelectFiles(templates), name="datasource")
datasource.inputs.base_directory = basedir

segment = Node(FAST(), name="segmentation")
segment.inputs.no_pve = True
segment.inputs.segments = True

def get_csf_file(tissue_class_files):
    print tissue_class_files
    return tissue_class_files[0]

meants = Node(ImageMeants(), name="meants")

def concat(means):
    pass

metaflow.connect([(infosource, datasource, [("subject_id", "subject_id")]),
                  (datasource, segment, [("struct", "in_files")]),
                  (datasource, meants, [("func", "in_file")]),
                  (segment, meants, [(("tissue_class_files", get_csf_file), "mask")]),
                  ])

#smooth = Node(Smooth(), name='smoothing')
#smooth.inputs.fwhm = 5.0

#metaflow.connect(datasource, "func", smooth, "in_file")

#tfilter = Node(TemporalFilter(), name="tfilter")
#tfilter.inputs.highpass_sigma = 1389

#metaflow.connect(smooth, "smoothed_file", tfilter, "in_file")

#gunzip = Node(Gunzip(), name="gunzip")

#metaflow.connect(tfilter, "out_file", gunzip, "in_file")

#conn = Node(FunctionalConnectivity(), name="conn")
#conn.inputs.atlas = os.path.join(basedir, "atlas", "aal_sampled.nii")

#metaflow.connect(gunzip, "out_file", conn, "fmri")

#def save_connectivity_data(basedir, normalized_matrix, regions):
    #print normalized_matrix
    #print regions
    ##import os
    ##import json
    ##regions_fname = os.path.join(
    ##with open(
    ##json.dumps(regions.tolist(),
    ##print regions
    ##print "ererer"

#save_conn = Node(Function(input_names=["basedir", "normalized_matrix", "regions"],
                          #output_names=["regions_file"],
                          #function=save_connectivity_data),
                 #name="save_conn")
#save_conn.inputs.basedir = basedir

#metaflow.connect(conn, "normalized_matrix", save_conn, "normalized_matrix")
#metaflow.connect(conn, "regions", save_conn, "regions")

#datasink = Node(DataSink(), name="sinker")
#datasink.inputs.base_directory = basedir + "/outputs"
#datasink.inputs.parameterization = False

#metaflow.connect(infosource, "subject_id", datasink, "container")
#metaflow.connect(tfilter, "out_file", datasink, "filtered")

##def print_output(val):
    ##print "here"
    ##print val
    ##return 0

##function = Node(Function(input_names=["val"], output_names="blub", function=print_output), name="function")

##metaflow.connect(datasource, "func", function, "val")

metaflow.run(plugin="Linear")

print "Finished"
