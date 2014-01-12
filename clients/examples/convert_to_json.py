import json
import csv


def parse_matrix_file(filename):
    matrix_data = []

    with open(filename, 'rb') as csvfile:
        matrix_reader = csv.reader(csvfile, delimiter=' ')
        for row in matrix_reader:
            row = map(lambda x: float(x), row)
            matrix_data.append(row)

    return matrix_data


def parse_regions_file(filename):
    regions_data = []

    with open(filename, 'rb') as csvfile:
        fieldnames = ('label', 'full_name', 'x', 'y', 'z', 'color')
        regions_reader = csv.DictReader(csvfile,
                                        delimiter=',',
                                        fieldnames=fieldnames)

        for row in regions_reader:
            regions_data.append(row)

        return regions_data


if __name__ == '__main__':
    matrix_data = parse_matrix_file("matrix.txt")
    with open('matrix.json', 'w') as outfile:
        json.dump(matrix_data, outfile)

    regions_data = parse_regions_file("regions.txt")
    with open('regions.json', 'w') as outfile:
        json.dump(regions_data, outfile)
