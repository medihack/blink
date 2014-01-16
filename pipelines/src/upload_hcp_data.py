#!/usr/bin/env python

import sys
import re
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
# scan subjects to process from provided file
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

if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
    print "Please provide a file with subject ids (one per line)."
    print "May be created with './manage_subjects -l path_to_subjects_folder'"
    sys.exit(2)

subjects = []

###
# Create and send network
###
with open(sys.argv[1]) as subjects_file:
    for line in subjects_file:
        line = re.sub(r"#.*", "", line) # remove comments
        line = line.strip()
        if not line:
            continue
        elif re.match(r"^\d", line):
            subjects.append(line)
        else:
            print "Invalid subject id: " + line
            sys.exit(2)

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
    request = client.Request(token, dev=True)

    result = request.create(network)
