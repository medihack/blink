#!/usr/bin/python

import sys
import getopt
import numpy
import csv
import json

# default parameters
params = {
    'size': 300,
    'format': "csv",
    'output': "matrix"
}


def create_symmetric_matrix(size):
    matrix = numpy.random.rand(size, size)
    for (i, j), v in numpy.ndenumerate(matrix):
        if i == j:  # make diagonal 0
            matrix[i][j] = 0
        elif i > j:  # make matrix symmetrical
            matrix[j][i] = matrix[i][j]

    return matrix


def save_csv(matrix, output):
    with open(output + ".txt", "w") as cvsfile:
        csvwriter = csv.writer(cvsfile, delimiter=" ")
        for row in matrix:
            csvwriter.writerow(row)


def save_json(matrix, output):
    with open(output + ".json", "w") as jsonfile:
        json.dump(matrix.tolist(), jsonfile)


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


if __name__ == '__main__':
    parse_params(sys.argv[1:], params)
    matrix = create_symmetric_matrix(params['size'])
    if params['format'] == "json":
        save_json(matrix, params['output'])
    elif params['format'] == "csv":
        save_csv(matrix, params['output'])
    else:
        print "invalid format"
        sys.exit(2)
