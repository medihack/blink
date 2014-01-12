classdef Request
    properties
        token
        url
        debug
    end
    
    methods
        function obj = Request(token, varargin)
            obj.token = token;
            
            if ~isempty(varargin)
                obj.debug = logical(varargin{1});
            else
                obj.debug = false;
            end
            
            if length(varargin) > 1
                dev = logical(varargin{2});
            else
                dev = false;
            end
            
            if (dev)
                obj.url = 'http://192.168.71.167:8000/blink/api/networks/';
            else
                obj.url = 'http://blink.neuromia.org/api/networks/';
            end
        end
        
        function networkId = create(obj, network)
            network.valid();
                
            h1 = http_createHeader('Content-Type', 'application/json');
            h2 = http_createHeader('Accept', 'application/json');
            h3 = http_createHeader('Authorization', strcat('Token', {' '}, obj.token));
            header = [h1 h2 h3];
            payload = network.to_json();
            
            [presponse, pstatus] = urlread2(obj.url, 'POST', payload, header);
            
            if pstatus.status.value ~= 201
                error(presponse);
            else
                network.id = str2num(presponse);
                networkId = network.id;
            end
        end
        
        function network = retrieve(obj, networkId)
            retrieve_url = strcat(obj.url, num2str(networkId), '/');
            header = http_createHeader('Accept', 'application/json');

            r = urlread2(retrieve_url, 'GET', '', header);
            
            data = loadjson(r);
            network = Network(data.title);
            fields = fieldnames(data);
            for i = 1:numel(fields)
                field = fields{i};
                value = data.(field);
                if ~isempty(value)
                    network.(field) = value;
                end
            end
        end
    end
end
