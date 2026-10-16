function P = khatrirao(varargin)

%% Error checking on input and set matrix order
% Note that this next if/else check forces A to be a cell array.
if ischar(varargin{end}) && varargin{end} == 'r'
    if nargin == 2 && iscell(varargin{1})
        % Input is a single cell array
        A = varargin{1};
    else
        % Input is a sequence of matrices
        A = {varargin{1:end-1}};
    end
    matorder = length(A):-1:1;
else
    if nargin == 1 && iscell(varargin{1})
        % Input is a single cell array
        A = varargin{1};
    else
        % Input is a sequence of matrices
        A = varargin;
    end
    matorder = 1:length(A);
end

%% Error check on matrices and compute number of rows in result 
ndimsA = cellfun(@ndims, A);
if(~all(ndimsA == 2))
    error('Each argument must be a matrix');
end

ncols = cellfun(@(x) size(x, 2), A);
if(~all(ncols == ncols(1)))
    error('All matrices must have the same number of columns.');
end


%% Computation
N = ncols(1);
P = A{matorder(1)};
for i = matorder(2:end)
    P = bsxfun(@times, reshape(A{i},[],1,N),reshape(P,1,[],N));
end
P = reshape(P,[],N);
