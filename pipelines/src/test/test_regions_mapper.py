#!/usr/bin/env python

import os
from blink_interface import FunctionalConnectivity

fmri_path = os.path.abspath("../workspace/fmri.nii")
atlas_path = os.path.abspath("../workspace/aal_sampled.nii")

conn = FunctionalConnectivity()
conn.inputs.fmri = fmri_path
conn.inputs.atlas = atlas_path
result = conn.run()
print result.outputs
