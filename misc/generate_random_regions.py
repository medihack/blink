#!/usr/bin/python

###
# Generates pseudo regions
###

import sys
import getopt
import csv
import json

# default parameters
params = {
    'size': 300,
    'format': "csv",
    'output': "regions"
}


def parse_params(argv, params):
    try:
        short_opts = "s:f:o:"
        long_opts = []
        opts, args = getopt.getopt(argv, short_opts, long_opts)

    except:
        print "Error! Invalid option(s)."
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-s":
            params['size'] = int(arg)
        elif opt == "-f":
            params['format'] = arg
        elif opt == "-o":
            params['output'] = arg


def create_regions(size):
    regions = []
    for i in range(size):
        region = {
            'label': "region" + str(i),
            'full_name': "Region " + str(i),
            'x': i,
            'y': i,
            'z': i
        }
        regions.append(region)

    return regions


def save_csv(regions, output):
    with open(output + ".txt", "w") as cvsfile:
        csvwriter = csv.writer(cvsfile, delimiter=",")
        for r in regions:
            row = [r['label'], r['full_name'], r['x'], r['y'], r['z']]
            csvwriter.writerow(row)


def save_json(regions, output):
    with open(output + ".json", "w") as jsonfile:
        json.dump(regions, jsonfile)


if __name__ == '__main__':
    parse_params(sys.argv[1:], params)
    regions = create_regions(params['size'])
    if params['format'] == "json":
        save_json(regions, params['output'])
    elif params['format'] == "csv":
        save_csv(regions, params['output'])
    else:
        print "invalid format"
        sys.exit(2)
