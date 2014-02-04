import sys
import re
import csv

###
# Scan subjects to process from the command line (space separated subject ids)
# or from a provided file (using the -f option, one subject id per line).
###
def get_subjects():
    if len(sys.argv) < 2:
        print "Please provide (space separated) subject ids to process."
        print "You may provide a file with subject ids (one per line) with the -f option."
        print "Such a file may be created with './manage_subjects -l path_to_subjects_folder'"
        print "Examples:"
        print sys.argv[0] + " 123093 329111 999323"
        print sys.argv[0] + " -f subjects.txt"
        sys.exit(2)

    subjects = []

    if sys.argv[1] == "-f":
        with open(sys.argv[2]) as subjects_file:
            for line in subjects_file:
                line = re.sub(r"#.*", "", line)  # remove comments
                line = line.strip()
                if not line:
                    continue
                elif re.match(r"^\d{6}$", line):
                    subjects.append(line)
                else:
                    print "Invalid subject id: " + line
                    sys.exit(2)
    else:
        del sys.argv[0]
        for subject_id in sys.argv:
            if re.match(r"^\d{6}$", subject_id):
                subjects.append(subject_id)
            else:
                print "Invalid subject id: " + subject_id
                sys.exit(2)

    return subjects

###
# Parse subjects data from CSV file (downloaded from Connectome DB).
# Returns a dictionary of the data by subject id.
###
def parse_subject_data(subjects_data_fname):
    subjects_data = dict()

    with open(subjects_data_fname) as sd_f:
        sd_reader = csv.reader(sd_f)
        for idx, row in enumerate(sd_reader):
            if idx == 0:
                continue  # skip the header

            subj_data = dict()

            subject_id = row[0].strip()
            if not re.match(r"^\d+$", subject_id):
                raise Exception("Invalid subject id: " + subject_id)

            gender = row[1].strip()
            if gender == "F":
                subj_data["gender"] = "female"
            elif gender == "M":
                subj_data["gender"] = "male"
            else:
                raise Exception("Invalid gender of subject: " + subject_id)

            age = row[2].strip()
            if re.match(r"^\d+-\d+$", age):
                subj_data["age"] = age
            elif re.match(r"^[<>]\d+$", age):
                subj_data["age"] = age
            else:
                raise Exception("Invalid age of subject: " + subject_id)

            subjects_data[subject_id] = subj_data

    return subjects_data
