#!/usr/bin/env python

import os
basedir = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(os.path.realpath(os.path.join(basedir, "..")))
from blink_interface import RegionsMapper

atlas_path = os.path.realpath(os.path.join(basedir, "..", "..", "data", "aal2mni_2mm.nii.gz"))
definitions_path = os.path.realpath(os.path.join(basedir, "..", "..", "data", "aal_regions_with_coords.txt"))

mapper = RegionsMapper()
mapper.inputs.regions = [1, 2, 3]
mapper.inputs.definitions = definitions_path
mapper.inputs.atlas = atlas_path
result = mapper.run()
print result.outputs
