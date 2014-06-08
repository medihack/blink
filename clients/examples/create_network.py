#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
folder = os.path.dirname(__file__)
sys.path.append(os.path.join(folder, "../src/python"))

from blink import client
import json

if __name__ == '__main__':
    # load matrix data from json file
    matrix_data = None
    matrix_file = os.path.join(folder, "matrix.json")
    with open(matrix_file) as infile:
        matrix_data = json.load(infile)

    # load regions data from json file
    regions_data = None
    regions_file = os.path.join(folder, "regions.json")
    with open(regions_file) as infile:
        regions_data = json.load(infile)

    # create network object
    network = client.Network("Example Subject")
    network.matrix = matrix_data
    network.regions = regions_data
    network.project = 'Human Connectome Project'
    network.atlas = 'AAL'
    network.subject_type = client.SubjectType.single
    network.gender = client.Gender.male
    network.age = 23
    network.scanner_device = 'Siemens Trio'
    network.scanner_parameters = 'TR 720ms, TE 33.1ms, Flip Angle 52°, ST 2.0mm iso, Multiband 8, BW 2290 Hz/Px'
    network.preprocessing = 'Preprocessing pipeline of Connectome Project, Smoothing 5mm Kernel'

    # create request object and upload network to BLINK server
    token = "2a367de02f52b927935cfa192422a2305eb3a087"
    request = client.Request(token)
    result = request.create(network)
    print "Created example network with ID: " + str(result)
