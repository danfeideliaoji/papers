function mode = cisf_ablation_mode(para)
%CISF_ABLATION_MODE Validate and canonicalize the optional CISF ablation.
%   Modes map to the paper's ablations as follows:
%   full                 : CISF
%   no_common            : CISFnC (remove W_C)
%   no_individual        : CISFnI (remove W_D)
%   no_label_correlation : CISFnL (fix S to the identity)

    mode = 'full';
    if isstruct(para) && isfield(para, 'ablation') && ~isempty(para.ablation)
        mode = para.ablation;
    end
    if isstring(mode)
        if ~isscalar(mode)
            error('CISF:InvalidAblation', 'ablation must be a character vector or string scalar.');
        end
        mode = char(mode);
    end
    if ~ischar(mode) || ~isrow(mode)
        error('CISF:InvalidAblation', 'ablation must be a character vector or string scalar.');
    end
    mode = lower(strtrim(mode));
    valid_modes = {'full', 'no_common', 'no_individual', 'no_label_correlation'};
    if ~any(strcmp(mode, valid_modes))
        error('CISF:InvalidAblation', ...
            'ablation must be full, no_common, no_individual, or no_label_correlation.');
    end
end
