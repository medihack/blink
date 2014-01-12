classdef Network
    properties  
        % general
        id = {};
        title
        matrix_data = {};
        regions_data = {};
        modality = Modality.unspecified;
        project = '';
        atlas = '';
        % subject
        subject_type = SubjectType.unspecified;
        group_size = {};
        gender = Gender.unspecified;
        age = {};
        age_mean = {};
        age_sd = {};
        pathology = '';
        %  protocol
        scanner_device = '';
        scanner_parameters = '';
        preprocessing = '';
        % misc
        funding = '';
        citation = '';
        note = '';
        % privacy
        private = false;
    end
    
    methods
        function obj = Network(title)
            obj.title = title;
        end
        
        function obj = addRegion(obj, label, full_name, x, y, z, varargin)
            region = struct();
            region.('label') = label;
            region.('full_name') = full_name;
            region.('x') = x;
            region.('y') = y;
            region.('z') = z;
            
            if ~isempty(varargin)
                region.('color') = varargin{1};
            end
            
            if length(varargin) > 1
                region.('note') = varargin{2};
            end
            
            Network.check_region(region);
            
            idx = length(obj.regions_data) + 1;
            obj.regions_data{idx} = region;
        end
        
        % do some basic validations (full validation done on server)
        function obj = valid(obj)
            if isempty(obj.title)
                error('missing title');
            end
                
            if isempty(obj.modality)
                error('missing modality');
            end
                
            if isempty(obj.matrix_data)
                error('missing matrix');
            end
                
            if isempty(obj.regions_data)
                error('missing regions');
            end
            
            for k=1:length(obj.regions_data)
                region = obj.regions_data{k};
                Network.check_region(region);
            end
        end
        
        function json = to_json(obj)
            serialized = struct(obj);
            json = savejson('', serialized, 'ArrayToStruct', 0, 'ParseLogical', 1, 'ForceRootName', 0);
        end
    end
    
    methods(Static)
        function check_region(region)
            if isempty(region.('label'))
                error('missing region label');
            elseif isempty(region.('full_name'))
                error('missing region full name');
            elseif isempty(region.('x')) || isempty(region.('y')) || isempty(region.('z'))
                error('missing region coordinates');
            end
        end
    end
end
