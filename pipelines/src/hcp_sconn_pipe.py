#!/usr/bin/env python

import os
import utils
from nipype.pipeline.engine import Node, MapNode, Workflow
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.fsl.dti import DTIFit, ProbTrackX
from nipype.interfaces.fsl.preprocess import ApplyWarp
from nipype.workflows.dmri.fsl.dti import create_bedpostx_pipeline
from nipype.interfaces.io import SelectFiles
from nipype.interfaces.utility import Function
from blink_interface import AtlasSplitter

###
# options
###
options = dict(
    workflow_plugin="SGE",
    number_of_processors=2,
    #sub_args="-q long.q -l h_vmem=15G,virtual_free=10G",
    sub_args="-q long.q",
    save_graph=False,
    debug=True
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
workflow = Workflow(name="hcp_sconn")
workflow.base_dir = os.path.join(basedir, "cache")

infosource = Node(IdentityInterface(fields=["subject_id"]),
                  name="infosource")
infosource.iterables = ("subject_id", subjects)

templates = dict(
    dwi="{subject_id}/T1w/Diffusion/data.nii.gz",
    mask="{subject_id}/T1w/Diffusion/nodif_brain_mask.nii.gz",
    bvals="{subject_id}/T1w/Diffusion/bvals",
    bvecs="{subject_id}/T1w/Diffusion/bvecs"
)
datasource = Node(SelectFiles(templates), name="datasource")
datasource.inputs.base_directory = os.path.join(basedir, "subjects")

dtifit = Node(DTIFit(), name="dtifit")

bedpostx = create_bedpostx_pipeline()

warp = Node(ApplyWarp(), name="warp")
warp.inputs.in_file = os.path.join(basedir, "data", "aal2mni_2mm.nii.gz")
warp.inputs.interp = "nn"

splitter = Node(AtlasSplitter(), name="splitter")

def calc_conn(seeds_to_target):
    print seeds_to_target
    return None

calcconn = Node(Function(input_names="targets",
                         output_names="matrix",
                         function=calc_conn),
                name="calcconn")

probtrackx = MapNode(ProbTrackX(), name='probtrackx', iterfield=["seed"])
probtrackx.inputs.os2t = True
#probtrackx.inputs.network = True
#probtrackx.inputs.mode = 'twomask_symm'

if options["debug"]:
    probtrackx.inputs.n_samples = 10  # default is 5000

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

                  # dtifit
                  (datasource, dtifit, [("dwi", "dwi")]),
                  (datasource, dtifit, [("mask", "mask")]),
                  (datasource, dtifit, [("bvals", "bvals")]),
                  (datasource, dtifit, [("bvecs", "bvecs")]),
                  (infosource, dtifit, [("subject_id", "base_name")]),

                  # bedpostx
                  (datasource, bedpostx, [("dwi", "inputnode.dwi")]),
                  (datasource, bedpostx, [("mask", "inputnode.mask")]),
                  (datasource, bedpostx, [("bvals", "inputnode.bvals")]),
                  (datasource, bedpostx, [("bvecs", "inputnode.bvecs")]),

                  # transform aal to dwi space
                  (datasource, warp, [("dwi", "ref_file")]),
                  (datasource, warp, [("mask", "mask_file")]),

                  # split atlas
                  (warp, splitter, [("out_file", "atlas")]),

                  # probtrackx
                  (datasource, probtrackx, [("mask", "mask")]),
                  (splitter, probtrackx, [("masks", "seed")]),
                  (splitter, probtrackx, [("masks", "target_masks")]),
                  (bedpostx, probtrackx, [("outputnode.thsamples", "thsamples"),
                                          ("outputnode.phsamples", "phsamples"),
                                          ("outputnode.fsamples", "fsamples")
                                          ]),

                  # calculate connectivity matrix
                  (probtrackx, calcconn, [("targets", "targets")])
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
