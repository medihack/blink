addpath(genpath('../../matlab/blink'));

token = '2a367de02f52b927935cfa192422a2305eb3a087';
req = Request(token, true, true);

load regions

load matrix_pat
network = Network('Stroke patients');
network.matrix_data = matrix_pat;
network.regions_data = regions;
networkId = req.create(network);
disp(networkId);

load matrix_con_hoax
network = Network('Age matched controls');
network.matrix_data = matrix_con_hoax;
network.regions_data = regions;
networkId = req.create(network);
disp(networkId);

load matrix_ttest2
network = Network('Stroke patients vs age matched controls');
network.matrix_data = matrix_ttest2;
network.regions_data = regions;
networkId = req.create(network);
disp(networkId);