function out = fastSimulatedMoments(paramOverrides, opt)
% FASTSIMULATEDMOMENTS  Simulate moments assuming agents never anticipate help.
%
%   OUT = FASTSIMULATEDMOMENTS(PARAMOVERRIDES, OPT) builds model objects,
%   solves the stationary no-help equilibrium, simulates the economy from the
%   initial distribution, and aggregates the requested moments. This routine is
%   streamlined for speed: it reuses the steady-state policies from NOHELPEQM
%   (i.e., agents believe they will never receive help) and performs a single
%   forward simulation over the desired horizon.
%
%   INPUTS
%   ------
%   paramOverrides : optional struct or numeric vector passed to SetParameters.
%   opt            : optional struct with runtime tweaks (all fields optional)
%                    .Nagents  number of simulated agents (default: 5000)
%                    .years    number of simulated years   (default: 7)
%                    .quarters total quarters to simulate  (overrides .years)
%                    .seed     RNG seed for reproducibility
%
%   OUTPUT
%   ------
%   out : struct with fields
%         .dims, .params, .settings
%         .crossSection   aggregate yearly cross-sections by location
%         .flows          newcomer composition shares
%         .cohorts        cohort-specific outcomes at requested horizons
%         .agentData      simulated trajectories (truncated to used quarters)
%         .flowLog        migration flow diagnostics used for moments
%
%   The model is quarterly; annual statistics aggregate four consecutive
%   quarters. Requested cohort statistics evaluate outcomes at specific years
%   for migrants grouped by their first year of arrival to the location.
%
% -------------------------------------------------------------------------

    if nargin < 1 || isempty(paramOverrides)
        paramOverrides = struct();
    end
    if nargin < 2 || isempty(opt)
        opt = struct();
    end

    %% 1) Dimensions, parameters, and settings ---------------------------------
    dims = setDimensionParam();

    settings.it      = 0;
    settings.diffV   = 1;
    settings.tolV    = 0.5;
    settings.tolM    = 1e-2;
    settings.MaxItV  = 40;
    settings.MaxItJ  = 10;
    settings.MaxIter = 100;

    settings.Nagents = 5000;
    defaultYears     = 7;   % need at least 7 years for requested cohorts
    quartersPerYear  = 4;
    settings.T       = quartersPerYear * defaultYears;

    if isfield(opt, 'Nagents') && ~isempty(opt.Nagents)
        settings.Nagents = max(1, round(opt.Nagents));
    end
    if isfield(opt, 'quarters') && ~isempty(opt.quarters)
        settings.T = max(1, round(opt.quarters));
    elseif isfield(opt, 'years') && ~isempty(opt.years)
        settings.T = quartersPerYear * max(1, round(opt.years));
    end

    if settings.T < quartersPerYear * defaultYears
        error('fastSimulatedMoments:shortHorizon', ...
            'Simulation horizon must cover at least %d years (>= %d quarters).', ...
            defaultYears, quartersPerYear * defaultYears);
    end

    if isfield(opt, 'seed') && ~isempty(opt.seed)
        rng(opt.seed);
        settings.seed = opt.seed;
    end

    params           = SetParameters(dims, paramOverrides);
    [grids, indexes] = setGridsAndIndices(dims);
    matrices         = constructMatrix(dims, params, grids, indexes);
    params.P         = matrices.P;

    %% 2) Steady-state policies under no help -----------------------------------
    [vf_nh, pol_nh] = noHelpEqm(dims, params, grids, indexes, matrices, settings); %#ok<NASGU>

    %% 3) Simulation with stationary no-help policies ---------------------------
    m0 = createInitialDistribution(dims, settings);

    G_noHelp = params.G0;
    if size(G_noHelp, 2) == 1
        G_dist = repmat(G_noHelp, 1, settings.T);
    else
        G_dist = G_noHelp;
        if size(G_dist, 2) < settings.T
            G_dist(:, end+1:settings.T) = repmat(G_dist(:, end), 1, settings.T - size(G_dist, 2));
        elseif size(G_dist, 2) > settings.T
            G_dist = G_dist(:, 1:settings.T);
        end
    end

    [~, ~, agentDataFull, flowLogFull] = simulateAgents( ...
        m0, pol_nh, G_dist, dims, params, grids, matrices, settings);

    %% 4) Aggregate quarterly data into annual moments --------------------------
    Tused = floor(settings.T / quartersPerYear) * quartersPerYear;
    agentData.location = agentDataFull.location(:, 1:Tused);
    agentData.state    = agentDataFull.state(:, 1:Tused);
    agentData.network  = agentDataFull.network(:, 1:Tused);
    agentData.skill    = agentDataFull.skill(:);
    agentData.wealth   = agentDataFull.wealth(:, 1:Tused);

    flowLog.helpUsed       = flowLogFull.helpUsed(:, 1:Tused);
    flowLog.directFromVzla = flowLogFull.directFromVzla(:, 1:Tused);

    moments = computeRequestedMoments(agentData, flowLog, dims, params, grids, quartersPerYear);

    %% 5) Pack outputs ----------------------------------------------------------
    out.dims        = dims;
    out.params      = params;
    out.settings    = settings;
    out.crossSection = moments.crossSection;
    out.flows        = moments.flows;
    out.cohorts      = moments.cohorts;
    out.moments      = moments;
    out.agentData    = agentData;
    out.flowLog      = flowLog;
end

%% Local helpers ================================================================
function moments = computeRequestedMoments(agentData, flowLog, dims, params, grids, quartersPerYear)

    locationTraj = double(agentData.location);
    stateTraj    = double(agentData.state);
    skillVec     = double(agentData.skill(:));

    [Nagents, T] = size(locationTraj);
    N = dims.N;
    numYears = floor(T / quartersPerYear);

    skillMat   = repmat(skillVec, 1, T);
    A_vals     = params.A(locationTraj);
    theta_idx  = sub2ind([dims.S, dims.N], skillMat, locationTraj);
    theta_vals = params.theta_s(theta_idx);

    psi_idx   = mod(stateTraj - 1, dims.k) + 1;
    psi_vals  = grids.psi(psi_idx);
    wage_vals = A_vals .* theta_vals .* (1 + psi_vals) .^ params.theta_k;

    isUnemployed = stateTraj <= dims.k;
    isEmployed   = ~isUnemployed;
    wage_vals(~isEmployed) = NaN;

    moved = false(Nagents, T);
    if T >= 2
        moved(:, 2:end) = locationTraj(:, 2:end) ~= locationTraj(:, 1:end-1);
    end

    helpUsed   = flowLog.helpUsed;
    directVzla = flowLog.directFromVzla;

    cross.avgIncome       = nan(N, numYears);
    cross.unemployment    = nan(N, numYears);
    flows.shareHelp       = nan(N, numYears);
    flows.shareNoHelp     = nan(N, numYears);
    flows.shareDirectVzla = nan(N, numYears);
    flows.shareIndirect   = nan(N, numYears);

    for y = 1:numYears
        qRange = (y-1) * quartersPerYear + (1:quartersPerYear);
        for loc = 1:N
            totalAgents = 0;
            totalUnemp  = 0;
            totalEmp    = 0;
            wageSum     = 0;
            arrivalsTot = 0;
            arrivalsHelp= 0;
            arrivalsDir = 0;

            for q = qRange
                locMask = (locationTraj(:, q) == loc);
                count_q = sum(locMask);
                if count_q > 0
                    totalAgents = totalAgents + count_q;
                    totalUnemp  = totalUnemp  + sum(isUnemployed(locMask, q));
                    empMask     = locMask & isEmployed(:, q);
                    empCount_q  = sum(empMask);
                    totalEmp    = totalEmp + empCount_q;
                    if empCount_q > 0
                        wageSum = wageSum + sum(wage_vals(empMask, q), 'omitnan');
                    end
                end

                if q > 1
                    arrivalMask = moved(:, q) & locMask;
                    arrivals_q  = sum(arrivalMask);
                    if arrivals_q > 0
                        arrivalsTot  = arrivalsTot  + arrivals_q;
                        arrivalsHelp = arrivalsHelp + sum(helpUsed(arrivalMask, q));
                        arrivalsDir  = arrivalsDir  + sum(directVzla(arrivalMask, q));
                    end
                end
            end

            if totalAgents > 0
                cross.unemployment(loc, y) = totalUnemp / totalAgents;
            end
            if totalEmp > 0
                cross.avgIncome(loc, y) = wageSum / totalEmp;
            end

            if arrivalsTot > 0
                shareHelp = arrivalsHelp / arrivalsTot;
                shareDir  = arrivalsDir  / arrivalsTot;
                flows.shareHelp(loc, y)       = shareHelp;
                flows.shareNoHelp(loc, y)     = 1 - shareHelp;
                flows.shareDirectVzla(loc, y) = shareDir;
                flows.shareIndirect(loc, y)   = 1 - shareDir;
            end
        end
    end

    %% Cohort preparation -------------------------------------------------------
    firstArrivalQuarter = nan(Nagents, N);
    for loc = 1:N
        arrivalMatrix = false(Nagents, T);
        arrivalMatrix(:, 1) = locationTraj(:, 1) == loc;
        if T >= 2
            arrivalMatrix(:, 2:end) = (locationTraj(:, 2:end) == loc) & ...
                                      (locationTraj(:, 1:end-1) ~= loc);
        end
        for agent = 1:Nagents
            firstQ = find(arrivalMatrix(agent, :), 1, 'first');
            if ~isempty(firstQ)
                firstArrivalQuarter(agent, loc) = firstQ;
            end
        end
    end
    firstArrivalYear = ceil(firstArrivalQuarter / quartersPerYear);

    %% Cohort-specific summaries -----------------------------------------------
    cohorts = struct();
    cohorts.loc2_year7 = cohortSummary(2, 7, 1:7, ...
        locationTraj, stateTraj, wage_vals, firstArrivalYear, dims, quartersPerYear);
    cohorts.loc3_year7 = cohortSummary(3, 7, 1:7, ...
        locationTraj, stateTraj, wage_vals, firstArrivalYear, dims, quartersPerYear);
    cohorts.loc4_year4 = cohortSummary(4, 4, 1:4, ...
        locationTraj, stateTraj, wage_vals, firstArrivalYear, dims, quartersPerYear);
    cohorts.loc5_year3 = cohortSummary(5, 3, 1:3, ...
        locationTraj, stateTraj, wage_vals, firstArrivalYear, dims, quartersPerYear);
    cohorts.loc5_year7_all = cohortSummary(5, 7, 1:7, ...
        locationTraj, stateTraj, wage_vals, firstArrivalYear, dims, quartersPerYear);
    cohorts.loc5_year7_first2 = cohortSummary(5, 7, 1:2, ...
        locationTraj, stateTraj, wage_vals, firstArrivalYear, dims, quartersPerYear);
    cohorts.loc6_year7_all = cohortSummary(6, 7, 1:7, ...
        locationTraj, stateTraj, wage_vals, firstArrivalYear, dims, quartersPerYear);
    cohorts.loc6_year7_first2 = cohortSummary(6, 7, 1:2, ...
        locationTraj, stateTraj, wage_vals, firstArrivalYear, dims, quartersPerYear);

    %% Pack outputs -------------------------------------------------------------
    cross.years = (1:numYears);

    moments.crossSection = cross;
    moments.flows        = flows;
    moments.cohorts      = cohorts;
    moments.meta.quartersPerYear = quartersPerYear;
    moments.meta.numYears        = numYears;
end

function summary = cohortSummary(locIdx, targetYear, cohortYears, ...
        locationTraj, stateTraj, wageVals, firstArrivalYear, dims, quartersPerYear)

    targetQuarter = targetYear * quartersPerYear;
    if targetQuarter > size(locationTraj, 2)
        error('cohortSummary:targetBeyondSimulation', ...
            'Requested year %d exceeds simulated horizon.', targetYear);
    end

    currentMask = (locationTraj(:, targetQuarter) == locIdx);
    stateNow    = stateTraj(:, targetQuarter);
    wageNow     = wageVals(:, targetQuarter);

    isUnemp = stateNow <= dims.k;
    isEmp   = ~isUnemp;

    cohortYears = cohortYears(:)';
    numCohorts  = numel(cohortYears);

    avgIncome = nan(numCohorts, 1);
    unempRate = nan(numCohorts, 1);

    for c = 1:numCohorts
        cohortMask = currentMask & (firstArrivalYear(:, locIdx) == cohortYears(c));
        if ~any(cohortMask)
            continue;
        end
        unempRate(c) = mean(isUnemp(cohortMask));
        empMask = cohortMask & isEmp;
        empCount = sum(empMask);
        if empCount > 0
            avgIncome(c) = sum(wageNow(empMask), 'omitnan') / empCount;
        end
    end

    combinedMask = currentMask & ismember(firstArrivalYear(:, locIdx), cohortYears);
    combinedUnemp = NaN;
    combinedIncome = NaN;
    if any(combinedMask)
        combinedUnemp = mean(isUnemp(combinedMask));
        empMask = combinedMask & isEmp;
        empCount = sum(empMask);
        if empCount > 0
            combinedIncome = sum(wageNow(empMask), 'omitnan') / empCount;
        end
    end

    summary.location     = locIdx;
    summary.targetYear   = targetYear;
    summary.cohortYears  = cohortYears;
    summary.avgIncome    = avgIncome;
    summary.unempRate    = unempRate;
    summary.combined.avgIncome = combinedIncome;
    summary.combined.unempRate = combinedUnemp;
end
