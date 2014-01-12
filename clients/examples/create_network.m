addpath(genpath('../matlab/blink'));

network = Network('Subject 1');

network.matrix_data = loadjson('matrix.json');

regions_data = loadjson('regions.json');
for k=1:length(regions_data)
    data = regions_data{k};
    label = data.label;
    fullName = data.full_name;
    x = str2num(data.x);
    y = str2num(data.y);
    z = str2num(data.z);
    network = network.addRegion(label, fullName, x, y, z);
end

token = '2a367de02f52b927935cfa192422a2305eb3a087';
req = Request(token, true, true);
networkId = req.create(network);
disp(networkId);