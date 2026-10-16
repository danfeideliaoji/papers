function split = make_cisf_split(n)
%MAKE_CISF_SPLIT Make one random 70/10/20 train/validation/test split.
%   The caller controls reproducibility with RNG.  A new RANDPERM call is
%   made on every invocation, so repeated calls provide independent splits.

    validateattributes(n, {'numeric'}, {'scalar', 'integer', 'positive'}, ...
        mfilename, 'n');

    n_train = floor(0.7 * n);
    n_validation = floor(0.1 * n);
    n_test = n - n_train - n_validation;
    if n_train == 0 || n_validation == 0 || n_test == 0
        error('CISF:SplitTooSmall', ...
            'At least 10 samples are required for a 70/10/20 split.');
    end

    permutation = randperm(n);
    split.train = permutation(1:n_train);
    split.validation = permutation(n_train + 1:n_train + n_validation);
    split.test = permutation(n_train + n_validation + 1:end);
end
