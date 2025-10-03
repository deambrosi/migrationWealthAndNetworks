function mom = computeSimulatedMoments(agentData, M_total, M_network, dims, params, grids, settings, matrices) %#ok<INUSD>
% COMPUTESIMULATEDMOMENTS  Build simulated moments from trajectories.
%
%   MOM = COMPUTESIMULATEDMOMENTS(AGENTDATA, M_TOTAL, M_NETWORK, DIMS, PARAMS,
%   GRIDS, SETTINGS, MATRICES) computes a set of arrays of moments which will
%   later be stacked into a vector for estimation. All moments are computed per
%   period t and/or by skill s and/or by location i, as specified below.
%
%   OUTPUT (mom): struct with fields
%     .unemp_rate_sit      [S×N×T]   unemployment rate by skill, location, time
%     .avg_wage_emp_sit    [S×N×T]   average employed income by skill, location, time
%     .mass_total_sit      [S×N×T]   total mass by skill, location, time (share of agents of skill s)
%     .share_new_help_sit  [S×N×T]   among new arrivals at (i,t,s): share with help (t=1 = NaN)
%     .share_new_direct_sit [S×N×T]  among new arrivals at (i,t,s): share direct from Venezuela (t=1 = NaN)
%     .tenure_unemp_id     {N}       each cell is [Dmax+1 × 1] unemp rate by tenure at T_i
%     .tenure_avg_wage_id  {N}       each cell is [Dmax+1 × 1] avg employed income by tenure at T_i
%     .Ti                  scalar    horizon used for tenure moments
%     .Dmax                scalar    tenure cap applied
% -------------------------------------------------------------------------

    S = dims.S;
    N = dims.N;

    locTraj   = double(agentData.location);   % [Nagents×T]
    stateTraj = double(agentData.state);      % [Nagents×T]
    skillVec  = double(agentData.skill(:));   % [Nagents×1]
    Nagents   = size(locTraj, 1);
    T         = size(locTraj, 2);

    if isfield(params, 'A_path') && size(params.A_path, 2) >= T
        t_index   = repmat(1:T, Nagents, 1);
        lin_idx   = sub2ind([N, T], locTraj(:), t_index(:));
        A_vals    = reshape(params.A_path(lin_idx), Nagents, T);
    else
        A_vals    = params.A(locTraj);
    end
    
    % Build [Nagents x T] skill matrix to match locTraj
    skillMat   = repmat(skillVec, 1, T);

    % Vectorized lookup of theta by (skill, location)
    theta_vals = params.theta_s(sub2ind([S, N], skillMat, locTraj));
    
    % Wages
    psi_idx    = mod(stateTraj - 1, dims.k) + 1;     % already in your file
    psi_vals   = grids.psi(psi_idx);
    wage_vals  = A_vals .* theta_vals .* (1 + psi_vals) .^ params.theta_k;
    
    % Only employed earnings matter
    isU        = stateTraj <= dims.k;
    isE        = ~isU;
    wage_vals(~isE) = NaN;  % only employed earnings matter for averages

    moved = false(Nagents, T);
    if T >= 2
        moved(:, 2:end) = locTraj(:, 2:end) ~= locTraj(:, 1:end-1);
    end

    hasFlowLog = isfield(agentData, 'flowLog');
    if hasFlowLog
        flowLog = agentData.flowLog;
        helpUsed   = logical(flowLog.helpUsed);
        directVzla = logical(flowLog.directFromVzla);
    else
        helpUsed   = false(Nagents, T);
        directVzla = false(Nagents, T);
    end

    %% Aggregate by skill/location/time --------------------------------------
    mom.unemp_rate_sit      = nan(S, N, T);
    mom.avg_wage_emp_sit    = nan(S, N, T);
    mom.mass_total_sit      = zeros(S, N, T);
    mom.share_new_help_sit   = nan(S, N, T);
    mom.share_new_direct_sit = nan(S, N, T);

    for s = 1:S
        skillMask    = (skillVec == s);
        totalSkill   = sum(skillMask);
        if totalSkill == 0
            mom.mass_total_sit(s, :, :) = NaN;
            continue;
        end

        loc_s   = locTraj(skillMask, :);
        isU_s   = isU(skillMask, :);
        wage_s  = wage_vals(skillMask, :);
        moved_s = moved(skillMask, :);
        help_s  = helpUsed(skillMask, :);
        direct_s= directVzla(skillMask, :);

        for t = 1:T
            loc_t = loc_s(:, t);
            for i = 1:N
                subset = (loc_t == i);
                count_i = sum(subset);
                if count_i == 0
                    mom.mass_total_sit(s, i, t) = 0;
                    if t >= 2
                        mom.share_new_help_sit(s, i, t)   = NaN;
                        mom.share_new_direct_sit(s, i, t) = NaN;
                    end
                    continue;
                end

                mom.mass_total_sit(s, i, t) = count_i / totalSkill;
                mom.unemp_rate_sit(s, i, t) = mean(isU_s(subset, t));
                mom.avg_wage_emp_sit(s, i, t) = mean(wage_s(subset, t), 'omitnan');

                if t >= 2
                    arrivals = subset & moved_s(:, t);
                    if any(arrivals)
                        mom.share_new_help_sit(s, i, t)   = mean(help_s(arrivals, t));
                        mom.share_new_direct_sit(s, i, t) = mean(direct_s(arrivals, t));
                    else
                        mom.share_new_help_sit(s, i, t)   = NaN;
                        mom.share_new_direct_sit(s, i, t) = NaN;
                    end
                end
            end
        end
    end

    if T >= 1
        mom.share_new_help_sit(:, :, 1)   = NaN;
        mom.share_new_direct_sit(:, :, 1) = NaN;
    end

    %% Tenure-cohort moments -------------------------------------------------
    if isfield(settings, 'Ti') && ~isempty(settings.Ti)
        Ti = min(max(1, settings.Ti), T);
    else
        Ti = T;
    end
    if isfield(settings, 'max_tenure') && ~isempty(settings.max_tenure)
        Dmax = max(0, min(settings.max_tenure, Ti - 1));
    else
        Dmax = max(0, Ti - 1);
    end

    tenureLevels = zeros(Nagents, 1);
    for n = 1:Nagents
        loc_now = locTraj(n, Ti);
        tenure  = 0;
        tau     = Ti - 1;
        while (tau >= 1) && (tenure < Dmax) && locTraj(n, tau) == loc_now
            tenure = tenure + 1;
            tau    = tau - 1;
        end
        tenureLevels(n) = tenure;
    end

    mom.tenure_unemp_id    = cell(N, 1);
    mom.tenure_avg_wage_id = cell(N, 1);

    unemp_Ti = isU(:, Ti);
    wage_Ti  = wage_vals(:, Ti);
    loc_Ti   = locTraj(:, Ti);

    for i = 1:N
        idx_i = (loc_Ti == i);
        if ~any(idx_i)
            mom.tenure_unemp_id{i}    = nan(Dmax + 1, 1);
            mom.tenure_avg_wage_id{i} = nan(Dmax + 1, 1);
            continue;
        end

        tenure_i = tenureLevels(idx_i);
        unemp_i  = unemp_Ti(idx_i);
        wage_i   = wage_Ti(idx_i);

        unemp_bins = nan(Dmax + 1, 1);
        wage_bins  = nan(Dmax + 1, 1);
        for d = 0:Dmax
            binMask = (tenure_i == d);
            if any(binMask)
                unemp_bins(d+1) = mean(unemp_i(binMask));
                wage_bins(d+1)  = mean(wage_i(binMask), 'omitnan');
            end
        end

        mom.tenure_unemp_id{i}    = unemp_bins;
        mom.tenure_avg_wage_id{i} = wage_bins;
    end

    mom.Ti   = Ti;
    mom.Dmax = Dmax;

    %% Optional: retain aggregate masses for diagnostics ---------------------
    mom.M_total   = M_total;
    mom.M_network = M_network;

end
