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
%         .M_total        total migrant shares by location over quarters
%         .M_network      networked migrant shares by location over quarters
%         .M              alias for .M_total (requested convenience field)
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

    [M_total, M_network, agentDataFull, flowLogFull] = simulateAgentsUpdatingG( ...
        m0, pol_nh, dims, params, grids, matrices, settings);

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
    out.M_total      = M_total;
    out.M_network    = M_network;
    out.M            = M_total;
end

%% Local helpers ================================================================
function [M_history, MIN_history, agentData, flowLog] = simulateAgentsUpdatingG( ...
        m0, pol, dims, params, grids, matrices, settings)

    T       = settings.T;
    Nagents = settings.Nagents;
    N       = dims.N;

    logFlows = nargout >= 4;

    % Ensure policy paths are cell arrays over time
    if ~iscell(pol.a)
        pol.a   = repmat({pol.a},  T-1, 1);
        pol.an  = repmat({pol.an}, T-1, 1);
        pol.mu  = repmat({pol.mu}, T-1, 1);
        pol.mun = repmat({pol.mun},T-1, 1);
    end

    P_local   = matrices.P;
    mig_costs = matrices.mig_costs;

    if isfield(matrices, 'Hbin')
        Hbin = matrices.Hbin;
    else
        Hbin = dec2bin(0:(size(mig_costs, 3) - 1), N) - '0';
    end
    numHelpStates = size(Hbin, 1);

    skillVec = double(reshape([m0.skill], [], 1));
    locCurr  = double(reshape([m0.location], [], 1));
    weaCurr  = double(reshape([m0.wealth], [], 1));
    staCurr  = double(reshape([m0.state], [], 1));
    netCurr  = double(reshape([m0.network], [], 1));

    if numel(skillVec) ~= Nagents
        error('simulateAgentsUpdatingG:badInitialDistribution', ...
            'Initial distribution length (%d) does not match Nagents (%d).', ...
            numel(skillVec), Nagents);
    end

    locationTraj = zeros(Nagents, T, 'uint16');
    wealthTraj   = zeros(Nagents, T, 'uint16');
    stateTraj    = zeros(Nagents, T, 'uint16');
    networkTraj  = zeros(Nagents, T, 'uint8');
    skillTraj    = uint16(skillVec);

    locationTraj(:, 1) = uint16(locCurr);
    wealthTraj(:, 1)   = uint16(weaCurr);
    stateTraj(:, 1)    = uint16(staCurr);
    networkTraj(:, 1)  = uint8(netCurr);

    if logFlows
        helpUsedTraj   = false(Nagents, T);
        directVzlaTraj = false(Nagents, T);
    else
        helpUsedTraj   = [];
        directVzlaTraj = [];
    end

    ggamma = 1;
    if isfield(params, 'ggamma') && ~isempty(params.ggamma)
        ggamma = params.ggamma;
    end

    numAh = numel(grids.ahgrid);
    numA  = numel(grids.agrid);

    for t = 1:(T-1)
        % --- Compute help distribution from current network masses ---
        netCounts = accumarray(locCurr, netCurr, [N, 1], @sum, 0);

        networkShare = zeros(N, 1);
        if Nagents > 0
            networkShare = netCounts / Nagents;
        end
        networkShare = max(0, min(1, networkShare));

        piRow = (networkShare.').^ggamma;  % 1 × N
        piRow = max(0, min(1, piRow));

        piTerm       = bsxfun(@power, piRow, Hbin);
        oneMinusTerm = bsxfun(@power, 1 - piRow, 1 - Hbin);
        helpWeights  = prod(piTerm .* oneMinusTerm, 2);

        nextLoc = zeros(Nagents, 1);
        nextWea = zeros(Nagents, 1);
        nextSta = zeros(Nagents, 1);
        nextNet = zeros(Nagents, 1);

        if logFlows
            helpCol   = helpUsedTraj(:, t+1);
            directCol = directVzlaTraj(:, t+1);
        end

        pol_a_t   = pol.a{t};
        pol_an_t  = pol.an{t};
        pol_mu_t  = pol.mu{t};
        pol_mun_t = pol.mun{t};
        helpDim_t = size(pol_mun_t, 6);
        if isempty(helpDim_t) || helpDim_t == 0
            helpDim_t = numHelpStates;
        end
        activeHelpDim = min(max(1, helpDim_t), numHelpStates);

        effectiveWeights = helpWeights(1:activeHelpDim);
        sumWeights = sum(effectiveWeights);
        if ~(sumWeights > 0) || ~isfinite(sumWeights)
            effectiveWeights = zeros(activeHelpDim, 1);
            effectiveWeights(1) = 1;
            sumWeights = 1;
        end
        effectiveDist = effectiveWeights / sumWeights;
        helpCdf = cumsum(effectiveDist);

        for agentIdx = 1:Nagents
            ski = skillVec(agentIdx);
            loc = locCurr(agentIdx);
            wea = weaCurr(agentIdx);
            sta = staCurr(agentIdx);
            net = netCurr(agentIdx);

            % Asset policy (fine grid index)
            if net >= 1
                a_fine = pol_an_t(ski, sta, wea, loc);
            else
                a_fine = pol_a_t(ski, sta, wea, loc);
            end
            a_fine = max(1, min(a_fine, numAh));
            a_fine = double(uint32(a_fine));

            [~, weaFineIdx] = min(abs(grids.agrid - grids.ahgrid(a_fine)));
            weaFineIdx = max(1, min(weaFineIdx, numA));
            nextWeaIdx = double(weaFineIdx);

            % Migration decision and help draw
            if net >= 1
                rHelp = rand();
                h_idx = find(helpCdf >= rHelp, 1, 'first');
                if isempty(h_idx)
                    h_idx = 1;
                end
                h_idx = min(max(1, h_idx), activeHelpDim);
                migProb = squeeze(pol_mun_t(ski, sta, nextWeaIdx, loc, :, h_idx));
            else
                h_idx = 1;
                migProb = squeeze(pol_mu_t(ski, sta, nextWeaIdx, loc, :));
            end

            if ~isvector(migProb) || numel(migProb) ~= N
                migProb = zeros(N, 1);
                migProb(loc) = 1;
            end
            migProb = double(migProb(:));
            probSum = sum(migProb);
            if ~(probSum > 0) || ~isfinite(probSum)
                migProb = zeros(N, 1);
                migProb(loc) = 1;
            else
                migProb = migProb / probSum;
            end

            cdfMig = cumsum(migProb);
            rMig   = rand();
            nextLocIdx = find(cdfMig >= rMig, 1, 'first');
            if isempty(nextLocIdx)
                [~, nextLocIdx] = max(migProb);
            end
            nextLocIdx = max(1, min(N, nextLocIdx));

            moved = (nextLocIdx ~= loc);
            helpFlag   = false;
            directFlag = false;

            if moved
                if logFlows && ~isempty(Hbin)
                    helpFlag = (net >= 1) && (Hbin(h_idx, nextLocIdx) == 1);
                end
                directFlag = (loc == 1);

                migCost = mig_costs(loc, nextLocIdx, min(max(1, h_idx), size(mig_costs, 3)));
                newAssets = grids.agrid(nextWeaIdx) - migCost;
                [~, newWeaIdx] = min(abs(grids.agrid - newAssets));
                newWeaIdx = max(1, min(newWeaIdx, numA));
                nextWeaIdx = double(newWeaIdx);
                sta = 1;
            else
                P_row = squeeze(P_local(ski, sta, :, loc));
                rowSum = sum(P_row);
                if rowSum > 0 && isfinite(rowSum)
                    P_row = P_row / rowSum;
                    cdfState = cumsum(P_row);
                    rState   = rand();
                    sta_next = find(cdfState >= rState, 1, 'first');
                    if isempty(sta_next)
                        sta_next = sta;
                    end
                    sta = sta_next;
                end
            end

            if net >= 1 && nextLocIdx ~= 1
                chi = params.cchi;
                if isscalar(chi)
                    pLose = chi;
                elseif isequal(size(chi), [dims.S, dims.N])
                    pLose = chi(ski, nextLocIdx);
                elseif isequal(size(chi), [1, dims.N])
                    pLose = chi(nextLocIdx);
                else
                    pLose = chi;
                end
                if rand() < pLose
                    net = 0;
                end
            end

            nextLoc(agentIdx) = nextLocIdx;
            nextWea(agentIdx) = nextWeaIdx;
            nextSta(agentIdx) = sta;
            nextNet(agentIdx) = net;

            if logFlows && moved
                helpCol(agentIdx)   = helpFlag;
                directCol(agentIdx) = directFlag;
            end
        end

        locCurr = nextLoc;
        weaCurr = nextWea;
        staCurr = nextSta;
        netCurr = nextNet;

        locationTraj(:, t+1) = uint16(locCurr);
        wealthTraj(:, t+1)   = uint16(weaCurr);
        stateTraj(:, t+1)    = uint16(staCurr);
        networkTraj(:, t+1)  = uint8(netCurr);

        if logFlows
            helpUsedTraj(:, t+1)   = helpCol;
            directVzlaTraj(:, t+1) = directCol;
        end
    end

    M_history   = zeros(N, T);
    MIN_history = zeros(N, T);
    for t = 1:T
        locs = double(locationTraj(:, t));
        nets = double(networkTraj(:, t));

        M_history(:, t) = accumarray(locs, 1, [N, 1], @sum, 0) / Nagents;
        netMask = (nets == 1);
        if any(netMask)
            MIN_history(:, t) = accumarray(locs(netMask), 1, [N, 1], @sum, 0) / Nagents;
        else
            MIN_history(:, t) = 0;
        end
    end

    agentData.location = locationTraj;
    agentData.wealth   = wealthTraj;
    agentData.state    = stateTraj;
    agentData.network  = networkTraj;
    agentData.skill    = skillTraj;

    if logFlows
        flowLog.helpUsed       = helpUsedTraj;
        flowLog.directFromVzla = directVzlaTraj;
    else
        flowLog = [];
    end
end

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
