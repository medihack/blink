About this repository
=====================

The BLINK GitHub repository contains the BLINK client (clients folder) to
communicate with the [BLINK website](http://neuromia.org/blink/) in
programmatic way using the REST API of the server.
Additionally the repository contains the pipeline we used to analyze one
hundred fMRI datasets from the [Human Connectome Project](http://www.humanconnectome.org/).

BLINK client introduction
=========================

The BLINK client allows to download and upload networks from resp. to the BLINK website in a programmatic way.
This is especially useful for batch processing of multiple networks.
We provide a Python and Matlab version for easy integration in existing workflows (like evaluations with FSL or SPM).


Python client example (with comments)
=====================================

+ Make sure `blink` is in the directory from where you use it or put it in your Python path
(pip installation is planned).

+ Import the blink client module.
`from blink import client`

+ Create a new network instance with a title (required), for example
`network = blink.Network('My Brain Network')`

+ The network has various attributes that describe its origin
 and how it was generated.
 For full description of all available (optional) attributes see
 section "Network Attributes" in this README.
`network.project = 'The next Human Connectome Project'`
`network.atlas = 'AAL'`

+ Add (required) matrix data (very simple here for demonstration).
For a full matrix format description see section "Matrix Format" in this README.
`network.matrix = [[0, 0.5], [0.5, 0]]`

+ Add (required) regions data (must match the matrix row size).
For full regions format description see section "Regions Format" in this README.
`network.add_region('LPG', 'Left Postcentral Gyrus', -40.91, -20.47, 53.66)`
`network.add_region('RPG', 'Right Postcentral Gyrus', 45.69, 9.53, 31.31)`

+ Create a new request instance with your token as authentication.
 You can get the token from the BLINK website when you are signed in, click on
 your username on the top left and choose the "Show/Change Token" link.
`token = '389439mysecrettoken93423'`
`request = blink.Request(token)`

+ Send and create the network on the BLINK server.
`request.create(network)`


Matlab client example (with Matlab comments)
============================================

+ Add the path (and all subfolders) of the Matlab client to the Matlab search path.
`addpath(genpath('../matlab'));`

+ Create a new network instance with a title (required).
`network = Network('My Brain Network')`

+ The network has various attributes that describe its origin
and how it was generated.
For full description of all available (optional) attributes see
section "Network Attributes" in this README.
`network.project = 'The next Human Connectome Project'`
`network.atlas = 'AAL'`

+ Add some (required) matrix data (very simple here for demonstration).
For full matrix format description see section "Matrix Format" in this README.
`network.matrix_data = [[0, 0.5], [0.5, 0]]`

License information
===================
We use the 3-clause BSD license; the full license is in the file LICENSE in the blink distribution.
Copyright (c) 2014, BLINK Developers All rights reserved.
