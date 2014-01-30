#!/usr/bin/env python

from nipype.pipeline.engine import Node, MapNode, Workflow
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataGrabber

workflow = Workflow(name="test_workflow")
workflow.base_dir = "/home/zeus/projects/blink/pipelines/workspace"

templates = dict(func="000000/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz")
datasource = Node(SelectFiles(templates), name="datasource")
datasource.inputs.base_directory = "/home/zeus/projects/blink/pipelines/workspace/subjects"
datasource.inputs.raise_on_empty = True

workflow.add_nodes([datasource])

workflow.config["execution"] = {"job_finished_timeout": 61}
workflow.run(
    plugin="SGE",
    plugin_args={
        "qsub_args": '-q long.q'
    }
)
