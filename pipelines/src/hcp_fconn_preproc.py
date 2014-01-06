#!/usr/bin/env python

import os
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.fsl.utils import Smooth
from nipype.interfaces.fsl.maths import TemporalFilter

basedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.join(basedir, "..", "workspace")
basedir = os.path.realpath(basedir)

metaflow = Workflow(name="hcp_fconn_preproc")
metaflow.base_dir = os.path.join(basedir, "cache")

infosource = Node(IdentityInterface(fields=["subject_id"]),
                  name="infosource")
infosource.iterables = ("subject_id", ["subject1"])

templates = dict(struct="{subject_id}/struct/T1w.nii.gz")
datasource = Node(SelectFiles(templates), name="datasource")
datasource.inputs.base_directory = basedir

metaflow.connect(infosource, "subject_id", datasource, "subject_id")

smooth = Node(Smooth(), name='smoothing')
smooth.inputs.fwhm = 5.0

metaflow.connect(datasource, "struct", smooth, "in_file")

tfilter = Node(TemporalFilter(), name="tfilter")

metaflow.connect(smooth, "smoothed_file", tfilter, "in_file")

datasink = Node(DataSink(), name="sinker")
datasink.inputs.base_directory = basedir + "/outputs"
datasink.inputs.parameterization = False

metaflow.connect(infosource, "subject_id", datasink, "container")
metaflow.connect(tfilter, "out_file", datasink, "filtered")

metaflow.run(plugin="Linear")

print "Finished"
