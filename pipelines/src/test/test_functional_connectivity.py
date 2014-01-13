#!/usr/bin/env python

import os
basedir = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(os.path.realpath(os.path.join(basedir, "..")))
from blink_interface import FunctionalConnectivity

fmri_path = os.path.join(basedir, "test_fmri.nii.gz")
atlas_path = os.path.realpath(os.path.join(basedir, "..", "..", "data", "aal2mni_2mm.nii.gz"))

conn = FunctionalConnectivity()
conn.inputs.fmri = fmri_path
conn.inputs.atlas = atlas_path
result = conn.run()
print result.outputs
