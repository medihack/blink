addpath(genpath('../matlab/blink'));

token = '2a367de02f52b927935cfa192422a2305eb3a087';
req = Request('token', true, true);
s = req.retrieve(1);