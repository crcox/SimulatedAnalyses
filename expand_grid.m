function XYZ = expand_grid(varargin)
    vals = cell(1,nargin);
    [vals{:}] = ndgrid(varargin{:});
    XYZ = cell2mat(cellfun(@(x) x(:), vals, 'UniformOutput',false));
end