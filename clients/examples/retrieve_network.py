#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
folder = os.path.dirname(__file__)
sys.path.append(os.path.join(folder, "../lib/python"))

from blink import blink

token = "2a367de02f52b927935cfa192422a2305eb3a087"
request = blink.Request(token, dev=True)
network = request.retrieve(1)
print network.__dict__
