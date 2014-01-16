import requests
import json


class Modality:
    unspecified = "unspecified"
    fmri = "fmri"
    dti = "dti"
    meg = "meg"
    eeg = "eeg"
    other = "other"


class SubjectType:
    unspecified = "unspecified"
    single = "single"
    multiple = "multiple"


class Gender:
    unspecified = "unspecified"
    male = "male"
    female = "female"
    mixed = "mixed"


class Network(object):
    def __init__(self, title):
        # general
        self.id = None
        self.title = title  # required
        self.matrix_data = None  # required
        self.regions_data = None  # required
        self.modality = Modality.unspecified
        self.project = ''
        self.atlas = ''
        # subject
        self.subject_type = SubjectType.unspecified
        self.group_size = None
        self.gender = Gender.unspecified
        self.age = None
        self.age_mean = None
        self.age_sd = None
        self.pathology = ''
        # protocol
        self.scanner_device = ''
        self.scanner_parameters = ''
        self.preprocessing = ''
        # misc
        self.funding = ''
        self.citation = ''
        self.note = ''
        # privacy
        self.private = False

    @property
    def matrix(self):
        return self.matrix_data

    @matrix.setter
    def matrix(self, data):
        self.matrix_data = data

    @property
    def regions(self):
        return self.regions_data

    @regions.setter
    def regions(self, data):
        self.regions_data = data

    def add_region(self, label, full_name, x, y, z, color, note):
        region = {
            'label': label,
            'full_name': full_name,
            'x': x,
            'y': y,
            'z': z
        }

        # optional
        if color:
            region['color'] = color

        if note:
            region['note'] = note

        self.check_region(region)

        self.regions.append(region)

    # do some basic validations (full validation on the server)
    def valid(self):
        if not self.title:
            raise Exception("missing title")

        if not self.modality:
            raise Exception("missing modality")

        if not self.matrix:
            raise Exception("missing matrix")

        if not self.regions:
            raise Exception("missing regions")

        for region in self.regions:
            self.check_region(region)

    def check_region(self, region):
        if not region['label']:
            raise Exception('missing region label')

        if not region['full_name']:
            raise Exception('missing region full name')

        if not (region['x'] and region['y'] and region['z']):
            raise Exception('missing region coordinates')

    def to_json(self):
        serialized = {}

        for attr, value in self.__dict__.iteritems():
            serialized[attr] = value

        return json.dumps(serialized)


class Request():
    def __init__(self, token, debug=False, dev=False):
        self._token = token

        if dev:
            self._url = "http://127.0.0.1:8000/blink/api/networks/"
        else:
            self._url = "http://blink.neuromia.org/api/networks/"

        self._debug = debug

    def create(self, network):
        network.valid()

        headers = {}
        headers['Content-Type'] = 'application/json'
        headers['Accept'] = 'application/json'
        headers['Authorization'] = "Token " + self._token
        payload = network.to_json()

        r = requests.post(self._url, data=payload, headers=headers)

        if r.status_code != requests.codes.created:
            raise Exception(r.text)
        else:  # success
            network.id = int(r.text)
            return network.id

    def retrieve(self, network_id):
        url = self._url + "%s/" % network_id
        headers = {'Accept': 'application/json'}
        r = requests.get(url, headers=headers)

        if r.status_code != requests.codes.ok:
            raise Exception(r.text)
        else:
            data = r.json()
            network = Network(data['title'])
            for field in data:
                value = data[field]
                if value:
                    setattr(network, field, value)
            return network
