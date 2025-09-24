function [mvec, map] = packMoments(mom, dims, settings)
% PACKMOMENTS  Stack structured moment arrays into a single column vector.
%
%   [MVEC, MAP] = PACKMOMENTS(MOM, DIMS, SETTINGS) linearizes selected fields
%   of MOM into one vector MVEC. MAP contains index ranges for each block.
%
%   Blocks (in order):
%     1) unemp_rate_sit       : vec of size S*N*T
%     2) avg_wage_emp_sit     : vec of size S*N*T
%     3) mass_total_sit       : vec of size S*N*T
%     4) share_new_help_sit   : vec of size S*N*(T-1)  (t>=2 only)
%     5) share_new_direct_sit : vec of size S*N*(T-1)
%     6) tenure (for each i): [unemp_by_tenure; avg_wage_by_tenure]
%
%   NaNs are converted to 0 in MVEC, and MAP.nan_mask stores positions of NaNs.
% -------------------------------------------------------------------------

    %#ok<INUSD>

    S = dims.S;
    N = dims.N;
    T = size(mom.unemp_rate_sit, 3);

    blocks     = cell(6, 1);
    block_info = cell(6, 1);
    idx_start  = 1;

    % Block 1: Unemployment rates
    blk = mom.unemp_rate_sit(:);
    blocks{1} = blk;
    block_info{1} = struct('name', 'unemp_rate_sit', 'start', idx_start, ...
        'stop', idx_start + numel(blk) - 1, 'shape', [S, N, T]);
    idx_start = block_info{1}.stop + 1;

    % Block 2: Average wages (employed)
    blk = mom.avg_wage_emp_sit(:);
    blocks{2} = blk;
    block_info{2} = struct('name', 'avg_wage_emp_sit', 'start', idx_start, ...
        'stop', idx_start + numel(blk) - 1, 'shape', [S, N, T]);
    idx_start = block_info{2}.stop + 1;

    % Block 3: Mass shares
    blk = mom.mass_total_sit(:);
    blocks{3} = blk;
    block_info{3} = struct('name', 'mass_total_sit', 'start', idx_start, ...
        'stop', idx_start + numel(blk) - 1, 'shape', [S, N, T]);
    idx_start = block_info{3}.stop + 1;

    % Block 4: Share of new arrivals with help (exclude t=1)
    if T >= 2
        blk_help = mom.share_new_help_sit(:, :, 2:end);
        blk_help = blk_help(:);
    else
        blk_help = [];
    end
    blocks{4} = blk_help;
    block_info{4} = struct('name', 'share_new_help_sit', 'start', idx_start, ...
        'stop', idx_start + numel(blk_help) - 1, 'shape', [S, N, max(T-1, 0)], ...
        't_range', 2:T);
    idx_start = block_info{4}.stop + 1;

    % Block 5: Share arriving directly from origin (exclude t=1)
    if T >= 2
        blk_dir = mom.share_new_direct_sit(:, :, 2:end);
        blk_dir = blk_dir(:);
    else
        blk_dir = [];
    end
    blocks{5} = blk_dir;
    block_info{5} = struct('name', 'share_new_direct_sit', 'start', idx_start, ...
        'stop', idx_start + numel(blk_dir) - 1, 'shape', [S, N, max(T-1, 0)], ...
        't_range', 2:T);
    idx_start = block_info{5}.stop + 1;

    % Block 6: Tenure stacks per location
    tenure_blocks = [];
    tenure_map    = cell(N, 1);
    Dmax = mom.Dmax;
    for i = 1:N
        ten_u = mom.tenure_unemp_id{i};
        ten_w = mom.tenure_avg_wage_id{i};
        if isempty(ten_u)
            ten_u = nan(Dmax + 1, 1);
        end
        if isempty(ten_w)
            ten_w = nan(Dmax + 1, 1);
        end
        stack_i = [ten_u(:); ten_w(:)];
        tenure_map{i} = struct('location', i, 'start', idx_start + numel(tenure_blocks), ...
            'stop', idx_start + numel(tenure_blocks) + numel(stack_i) - 1, ...
            'length', numel(stack_i));
        tenure_blocks = [tenure_blocks; stack_i]; %#ok<AGROW>
    end
    blocks{6} = tenure_blocks;
    block_info{6} = struct('name', 'tenure_blocks', 'start', idx_start, ...
        'stop', idx_start + numel(tenure_blocks) - 1, ...
        'details', tenure_map);
    idx_start = block_info{6}.stop + 1;

    % Concatenate blocks
    fullvec = vertcat(blocks{:});
    nan_mask = isnan(fullvec);
    mvec = fullvec;
    mvec(nan_mask) = 0;

    map.block_info = block_info;
    map.nan_mask   = nan_mask;
    map.length     = numel(mvec);
    map.order      = {'unemp_rate_sit', 'avg_wage_emp_sit', 'mass_total_sit', ...
                      'share_new_help_sit', 'share_new_direct_sit', 'tenure_blocks'};
    map.tenure     = tenure_map;
    map.S          = S;
    map.N          = N;
    map.T          = T;
    if isstruct(settings) && isfield(settings, 'T')
        map.settings_T = settings.T;
    else
        map.settings_T = T;
    end

end
