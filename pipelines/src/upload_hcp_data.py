#!/usr/bin/env python

import utils
import os
import json
from blink import client

###
# setup basedir
###
basedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.join(basedir, "..", "workspace", "outputs")
basedir = os.path.realpath(basedir)

###
# scan subjects to process
###
subjects = utils.get_subjects()

###
# create network
###
def create_network(network_data, matrix_data, regions_data):
    network = client.Network(network_data.pop("title"))

    for key, value in network_data.items():
        setattr(network, key, value)

    network.matrix = matrix_data

    for region_data in regions_data:
        network.add_region(
            region_data["label"],
            region_data["full_name"],
            region_data["x"],
            region_data["y"],
            region_data["z"]
        )

    return network

###
# iterate subjects and send networks
###
for subject in subjects:
    print "Uploading subject: " + subject

    folder = os.path.join(basedir, subject, "rfMRI_Rest1_LR")

    if not os.path.isdir(folder):
        raise Exception("Invalid subject folder: " + subject)

    network_fname = os.path.join(folder, "network.json")
    with open(network_fname) as f:
        network_data = json.load(f)

    matrix_fname = os.path.join(folder, "matrix.json")
    with open(matrix_fname) as f:
        matrix_data = json.load(f)

    regions_fname = os.path.join(folder, "regions.json")
    with open(regions_fname) as f:
        regions_data = json.load(f)

    network = create_network(network_data, matrix_data, regions_data)

    token = "2a367de02f52b927935cfa192422a2305eb3a087"
    request = client.Request(token)

    result = request.create(network)
