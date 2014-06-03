#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
folder = os.path.dirname(__file__)
sys.path.append(os.path.join(folder, "../src/python"))

from blink import blink
import json

if __name__ == '__main__':
    matrix_data = None
    matrix_file = os.path.join(folder, "matrix.json")
    with open(matrix_file) as infile:
        matrix_data = json.load(infile)

    regions_data = None
    regions_file = os.path.join(folder, "regions.json")
    with open(regions_file) as infile:
        regions_data = json.load(infile)

    token = "2a367de02f52b927935cfa192422a2305eb3a087"
    request = blink.Request(token, dev=True)

    subject = "Subject 1"
    network = blink.Network(subject)
    network.matrix = matrix_data
    network.regions = regions_data

    network.project = 'Human Connectome Project'
    network.atlas = 'AAL'

    # subject
    network.subject_type = blink.SubjectType.single
    network.gender = blink.Gender.male
    network.age = 23

    # protocol
    network.scanner_device = 'Siemens Trio'
    network.scanner_parameters = 'TR 720ms, TE 33.1ms, Flip Angle 52°, ST 2.0mm iso, Multiband 8, BW 2290 Hz/Px'
    network.preprocessing = 'Preprocessing pipeline of Connectome Project, Smoothing 5mm Kernel'

    result = request.create(network)
    print result