function [M_history, MIN_history, agentData, flowLog, aidLog] = simulateAgentsWithVirtualAid(m0, pol, G_dist, dims, params, grids, matrices, settings, aidOpt)
% SIMULATEAGENTSWITHVIRTUALAID  Simulate dynamics with a virtual aid program.
%
%   This routine mirrors utils/simulateAgents but layers in an external
%   program that artificially increases the probability that migrants obtain
%   relocation help.  The program draws an additional help vector h' and
%   merges it with the baseline help draw h (elementwise max) so that any new
%   destinations unlocked by the program become available.  If the resulting
%   migration choice differs from the baseline choice, we interpret the move
%   as subsidised by the program and deduct (1 - alpha) * tau^{i\ell} from the
%   aid budget.  Each agent can benefit from the program at most once and the
%   budget is evaluated after every period; once funds are depleted the
%   program is switched off for the remaining periods.
%
%   INPUTS
%       m0        : struct array with initial agent states (location, wealth,
%                   state, network, skill).
%       pol       : policy functions returned by the equilibrium solver.
%       G_dist    : distribution over help configurations by period.
%       dims      : dimension structure (numbers of locations, skills, etc.).
%       params    : parameter structure (contains alpha, tau, chi, etc.).
%       grids     : asset grids used in the policies.
%       matrices  : precomputed matrices (transition matrix, migration costs,
%                   helper configuration binary matrix Hbin, ...).
%       settings  : simulation settings (Nagents, T, burn, ...).
%       aidOpt    : optional struct configuring the aid program with fields
%                     .totalBudget   Total funds available for subsidies.
%                     .startPeriod   First decision period when aid is active.
%                     .virtualMass   Scalar or vector added to G_dist before
%                                    drawing the program help h'.
%                     .shuffleAgents Whether to reshuffle agents initially.
%                     .name          Label stored in the returned aidLog.
%
%   OUTPUTS
%       M_history   : aggregate location shares over time.
%       MIN_history : location shares for agents with network access.
%       agentData   : struct with individual trajectories (location, wealth,
%                     employment state, network status, skill).
%       flowLog     : struct with boolean masks tracking help usage.
%       aidLog      : detailed accounting for the virtual aid program.

    % ------------------------------------------------------------------
    % (1) Sanitize optional configuration and set defaults.
    % ------------------------------------------------------------------
    if nargin < 9 || isempty(aidOpt)
        aidOpt = struct();                          % Use an empty struct when options omitted.
    end
    if ~isfield(aidOpt, 'totalBudget'), aidOpt.totalBudget = 0; end
    if ~isfield(aidOpt, 'startPeriod'), aidOpt.startPeriod = 1; end
    if ~isfield(aidOpt, 'virtualMass'), aidOpt.virtualMass = 0; end
    if ~isfield(aidOpt, 'shuffleAgents'), aidOpt.shuffleAgents = true; end
    if ~isfield(aidOpt, 'name'), aidOpt.name = 'VirtualAid'; end

    totalBudget   = max(0, double(aidOpt.totalBudget));      % Non-negative total resources.
    startPeriod   = max(1, round(double(aidOpt.startPeriod)));% First period when aid can operate.
    shuffleAgents = logical(aidOpt.shuffleAgents);            % Convert shuffle flag to logical.
    programName   = char(aidOpt.name);                        % Store descriptive label.

    % Interpret the virtualMass option as a vector with one entry per help state.
    if isscalar(aidOpt.virtualMass)
        virtualMass = repmat(double(aidOpt.virtualMass), numel(G_dist(:, 1)), 1);
    else
        virtualMass = double(aidOpt.virtualMass(:));
        if numel(virtualMass) ~= size(G_dist, 1)
            error('simulateAgentsWithVirtualAid:virtualMassSize', ...
                'virtualMass must be scalar or length matching rows of G_dist.');
        end
    end

    % ------------------------------------------------------------------
    % (2) Extract frequently used sizes and helper encodings.
    % ------------------------------------------------------------------
    T       = settings.T;                   % Number of simulation periods.
    Nagents = settings.Nagents;             % Number of agents in the simulated population.
    N       = dims.N;                       % Number of geographic locations.

    if isfield(matrices, 'Hbin')
        Hbin = matrices.Hbin;               % Binary encoding of helper availability per destination.
    else
        % Fallback: enumerate all binary combinations based on migration cost tensor size.
        Hbin = dec2bin(0:(size(matrices.mig_costs, 3) - 1), N) - '0';
    end
    numHelpStates = size(Hbin, 1);          % Total number of helper configurations encoded in Hbin.

    % Pre-compute a lookup from binary help vectors to their indices for quick matching.
    helpKey = containers.Map('KeyType', 'char', 'ValueType', 'uint32');
    for idx = 1:numHelpStates
        key = char('0' + Hbin(idx, :));     % Convert binary row into a character key (e.g., '010').
        helpKey(key) = uint32(idx);         % Store the corresponding index for retrieval later.
    end

    % ------------------------------------------------------------------
    % (3) Cache policy tensors, grids, and matrices for readability.
    % ------------------------------------------------------------------
    polA   = pol.a;                         % Policy for assets when no network help is active.
    polAn  = pol.an;                        % Asset policy when network help is active.
    polMu  = pol.mu;                        % Migration probabilities without help.
    polMun = pol.mun;                       % Migration probabilities with help configuration index.

    % Expand stationary policies across time so we can index with {t} safely.
    if ~iscell(polA)
        polA   = repmat({polA},  T-1, 1);
        polAn  = repmat({polAn}, T-1, 1);
        polMu  = repmat({polMu},  T-1, 1);
        polMun = repmat({polMun},T-1, 1);
    end

    agrid   = grids.agrid;                  % Asset grid corresponding to state indices.
    ahgrid  = grids.ahgrid;                 % Asset choice grid produced by the policy.
    Ptrans  = matrices.P;                   % Employment transition probabilities.
    migCost = matrices.mig_costs;           % Migration cost tensor by origin, destination, help index.
    tau     = params.ttau;                  % Baseline migration costs used for subsidy accounting.
    alpha   = params.aalpha;                % Fraction of the migration cost paid by the migrant when helped.

    % ------------------------------------------------------------------
    % (4) Allocate trajectory storage and initialise with period-1 states.
    % ------------------------------------------------------------------
    locationTraj = zeros(Nagents, T, 'uint16');
    wealthTraj   = zeros(Nagents, T, 'uint16');
    stateTraj    = zeros(Nagents, T, 'uint16');
    networkTraj  = zeros(Nagents, T, 'uint8');
    skillTraj    = zeros(Nagents, 1, 'uint16');

    % Boolean masks used for flow logs (help usage and program-specific help).
    helpUsedMask    = false(Nagents, T);
    programHelpMask = false(Nagents, T);

    % Populate period-1 states directly from the initial condition struct.
    for agentIdx = 1:Nagents
        locationTraj(agentIdx, 1) = uint16(m0(agentIdx).location);
        wealthTraj(agentIdx, 1)   = uint16(m0(agentIdx).wealth);
        stateTraj(agentIdx, 1)    = uint16(m0(agentIdx).state);
        networkTraj(agentIdx, 1)  = uint8(m0(agentIdx).network);
        skillTraj(agentIdx, 1)    = uint16(m0(agentIdx).skill);
    end

    % ------------------------------------------------------------------
    % (5) Set up accounting variables for the budget and program usage.
    % ------------------------------------------------------------------
    perPeriodSpent      = zeros(1, T);      % Subsidy dollars consumed in each decision period.
    perPeriodRecipients = zeros(1, T);      % Number of agents whose decision changed due to the program.
    cumulativeSpent     = 0;                % Running total of dollars spent so far.
    programEverUsed     = false(Nagents, 1);% Track which agents have already benefited from the program.

    % If requested, randomise the order of agents before the first period.
    if shuffleAgents
        baseOrder = randperm(Nagents);
    else
        baseOrder = 1:Nagents;
    end

    % The agent order remains fixed across periods to keep trajectories consistent.
    agentOrder = baseOrder(:)';

    % Budget path initialised at full resources in period 1.
    budgetPath = zeros(1, T);
    budgetPath(1) = totalBudget;

    % ------------------------------------------------------------------
    % (6) Iterate over decision periods (outer loop) then agents (inner loop).
    % ------------------------------------------------------------------
    for t = 1:(T - 1)
        % Determine whether the program is active this period (budget left and past the start).
        programActive = (t >= startPeriod) && (cumulativeSpent < totalBudget);

        % Temporary storage for program usage during this specific period.
        programUseThisPeriod  = false(Nagents, 1);
        subsidyThisPeriod     = zeros(Nagents, 1);

        % Loop over agents in the predetermined order.
        for orderIdx = 1:Nagents
            agentIdx = agentOrder(orderIdx);                % Actual agent index processed now.

            % Fetch current state variables for the agent at the beginning of period t.
            loc = locationTraj(agentIdx, t);               % Current location index.
            wea = wealthTraj(agentIdx, t);                 % Current wealth index.
            sta = stateTraj(agentIdx, t);                  % Current employment state index.
            net = networkTraj(agentIdx, t);                % Current network indicator (0/1).
            ski = skillTraj(agentIdx);                     % Skill type (time-invariant).

            % ------------------------------------------------------------------
            % (6a) Asset accumulation decision: map optimal savings back to the state grid.
            % ------------------------------------------------------------------
            if net == 1
                assetChoiceIdx = polAn{t}(ski, sta, wea, loc); % Use policy with network help.
            else
                assetChoiceIdx = polA{t}(ski, sta, wea, loc);  % Use policy without network help.
            end
            assetChoiceIdx = uint32(min(max(assetChoiceIdx, 1), numel(ahgrid))); % Clamp to grid bounds.

            % Translate the chosen asset level (from ahgrid) to the closest index on agrid.
            targetAsset = grids.ahgrid(assetChoiceIdx);        % Continuous asset chosen by the policy.
            [~, nextWealthIdx] = min(abs(agrid - targetAsset));% Locate the nearest discrete state.
            nextWealthIdx = uint16(max(1, min(nextWealthIdx, numel(agrid))));

            % ------------------------------------------------------------------
            % (6b) Baseline helper draw h and corresponding migration probabilities.
            % ------------------------------------------------------------------
            if net == 1
                % Retrieve and normalise the baseline helper mass for period t.
                helperMass = double(G_dist(:, t));
                helperMass(helperMass < 0) = 0;              % Ensure no negative weights.
                totalMass = sum(helperMass);
                if totalMass <= 0
                    helperMass = ones(numHelpStates, 1);    % Fallback to uniform when mass missing.
                    totalMass = numHelpStates;
                end
                helperProb = helperMass / totalMass;         % Convert to a probability distribution.

                % Draw baseline helper state using an inverse-CDF draw.
                uHelpBaseline = rand();
                cdfHelp = cumsum(helperProb);
                hIdxBaseline = find(cdfHelp >= uHelpBaseline, 1, 'first');
                if isempty(hIdxBaseline)
                    hIdxBaseline = 1;                        % Safety net when numerical issues arise.
                end
                hIdxBaseline = uint16(hIdxBaseline);
                helpVecBaseline = Hbin(double(hIdxBaseline), :); % Binary help availability vector.
            else
                % Without a network, the baseline help vector is identically zero.
                hIdxBaseline = uint16(1);
                helpVecBaseline = zeros(1, N);
            end

            % ------------------------------------------------------------------
            % (6c) Program help draw h' and merged help vector h_final.
            % ------------------------------------------------------------------
            hIdxFinal = hIdxBaseline;                       % Default to baseline configuration.
            helpVecFinal = helpVecBaseline;                 % Copy baseline help vector.

            % Only attempt to use the program when active, the agent has not benefited before,
            % and the agent currently has network access (policies with help exist only then).
            programEligible = programActive && (net == 1) && ~programEverUsed(agentIdx);
            if programEligible
                % Inflate the helper mass by adding the virtualMass vector before drawing h'.
                helperMassProgram = helperMass + virtualMass;
                helperMassProgram(helperMassProgram < 0) = 0; % Guard against negative adjustments.
                totalMassProgram = sum(helperMassProgram);
                if totalMassProgram <= 0
                    helperMassProgram = ones(numHelpStates, 1);
                    totalMassProgram = numHelpStates;
                end
                helperProbProgram = helperMassProgram / totalMassProgram;

                % Draw the program's helper configuration.
                uHelpProgram = rand();
                cdfHelpProgram = cumsum(helperProbProgram);
                hIdxProgram = find(cdfHelpProgram >= uHelpProgram, 1, 'first');
                if isempty(hIdxProgram)
                    hIdxProgram = 1;
                end
                helpVecProgram = Hbin(double(hIdxProgram), :);

                % Merge the baseline and program help vectors elementwise.
                helpVecFinal = max(helpVecBaseline, helpVecProgram);

                % Locate the index corresponding to the merged help vector so we can query policies.
                helpKeyFinal = char('0' + helpVecFinal);
                if isKey(helpKey, helpKeyFinal)
                    hIdxFinal = uint16(helpKey(helpKeyFinal));
                else
                    error('simulateAgentsWithVirtualAid:MissingHelpIndex', ...
                        'Merged help vector not found in Hbin lookup table.');
                end
            end

            % ------------------------------------------------------------------
            % (6d) Migration decision under baseline and program help configurations.
            % ------------------------------------------------------------------
            % Baseline migration probabilities given h.
            if net == 1
                migProbBaseline = squeeze(polMun{t}(ski, sta, nextWealthIdx, loc, :, hIdxBaseline));
            else
                migProbBaseline = squeeze(polMu{t}(ski, sta, nextWealthIdx, loc, :));
            end
            migProbBaseline = reshape(migProbBaseline, [N, 1]);
            totalProbBase = sum(migProbBaseline);
            if totalProbBase <= 0 || ~isfinite(totalProbBase)
                migProbBaseline = zeros(N, 1);               % Degenerate fallback: stay put.
                migProbBaseline(loc) = 1;
                totalProbBase = 1;
            end
            migProbBaseline = migProbBaseline / totalProbBase; % Normalise to sum to one.

            % Program-adjusted migration probabilities (equal to baseline when no program).
            if net == 1
                migProbFinal = squeeze(polMun{t}(ski, sta, nextWealthIdx, loc, :, hIdxFinal));
            else
                migProbFinal = squeeze(polMu{t}(ski, sta, nextWealthIdx, loc, :));
            end
            migProbFinal = reshape(migProbFinal, [N, 1]);
            totalProbFinal = sum(migProbFinal);
            if totalProbFinal <= 0 || ~isfinite(totalProbFinal)
                migProbFinal = zeros(N, 1);
                migProbFinal(loc) = 1;
                totalProbFinal = 1;
            end
            migProbFinal = migProbFinal / totalProbFinal;

            % Use the same uniform draw for both baseline and final migration decisions.
            uMove = rand();
            cdfBaseline = cumsum(migProbBaseline);
            baselineDest = find(cdfBaseline >= uMove, 1, 'first');
            if isempty(baselineDest)
                baselineDest = double(loc);
            end
            baselineDest = uint16(baselineDest);

            cdfFinal = cumsum(migProbFinal);
            finalDest = find(cdfFinal >= uMove, 1, 'first');
            if isempty(finalDest)
                finalDest = double(loc);
            end
            finalDest = uint16(finalDest);

            % Detect whether the program altered the migration decision.
            programUsedNow = programEligible && (finalDest ~= baselineDest);

            % ------------------------------------------------------------------
            % (6e) Apply migration, update wealth/state/network, and record logs.
            % ------------------------------------------------------------------
            moved = (finalDest ~= loc);
            helpAvailableFinal = moved && (helpVecFinal(double(finalDest)) == 1);

            if moved
                % Determine migration cost under the final help configuration.
                migCostActual = migCost(double(loc), double(finalDest), double(hIdxFinal));
                newAssets = agrid(nextWealthIdx) - migCostActual;            % Assets after paying migration cost.
                [~, adjWealthIdx] = min(abs(agrid - newAssets));             % Project back to the grid.
                nextWealthIdx = uint16(max(1, min(adjWealthIdx, numel(agrid))));

                % After moving, employment state resets to 1 (as in baseline simulator).
                nextState = uint16(1);
            else
                % No move: sample next employment state from the transition matrix.
                transRow = squeeze(Ptrans(ski, sta, :, loc));
                rowSum = sum(transRow);
                if rowSum > 0 && isfinite(rowSum)
                    transRow = transRow / rowSum;
                    cdfTrans = cumsum(transRow);
                    uTrans = rand();
                    nextState = find(cdfTrans >= uTrans, 1, 'first');
                    if isempty(nextState)
                        nextState = double(sta);
                    end
                else
                    nextState = double(sta);
                end
                nextState = uint16(nextState);
            end

            % Update the agent's network status: potential loss when moving abroad.
            nextNetwork = net;
            if net == 1 && moved && finalDest ~= 1
                chi = params.cchi;
                if isscalar(chi)
                    loseProb = chi;
                elseif isequal(size(chi), [dims.S, dims.N])
                    loseProb = chi(ski, finalDest);
                elseif isequal(size(chi), [1, dims.N])
                    loseProb = chi(finalDest);
                else
                    loseProb = chi;
                end
                if rand() < loseProb
                    nextNetwork = uint8(0);
                end
            end
            nextNetwork = uint8(nextNetwork);

            % Store period t+1 states back into the trajectory arrays.
            locationTraj(agentIdx, t + 1) = finalDest;
            wealthTraj(agentIdx, t + 1)   = nextWealthIdx;
            stateTraj(agentIdx, t + 1)    = nextState;
            networkTraj(agentIdx, t + 1)  = nextNetwork;

            % Record whether help (baseline or program) was used in this decision.
            helpUsedMask(agentIdx, t + 1) = helpAvailableFinal;
            programHelpMask(agentIdx, t + 1) = programUsedNow;

            % Keep track of program usage to enforce the "once per agent" rule.
            if programUsedNow
                programEverUsed(agentIdx) = true;
                programUseThisPeriod(agentIdx) = true;

                % The subsidy equals the share of migration cost paid by the program.
                subsidyAmount = (1 - alpha) * double(tau(double(loc), double(finalDest)));
                subsidyThisPeriod(agentIdx) = subsidyAmount;
            end
        end % agent loop

        % Sum period spending and recipients, then update the running totals.
        periodSpent = sum(subsidyThisPeriod);
        periodRecipients = sum(programUseThisPeriod);
        perPeriodSpent(t + 1) = periodSpent;
        perPeriodRecipients(t + 1) = periodRecipients;

        cumulativeSpent = cumulativeSpent + periodSpent;
        budgetRemaining = max(0, totalBudget - cumulativeSpent);
        budgetPath(t + 1) = budgetRemaining;
    end % time loop

    % ------------------------------------------------------------------
    % (7) Build aggregate location histories (total population and with network).
    % ------------------------------------------------------------------
    M_history = zeros(N, T);
    MIN_history = zeros(N, T);
    for t = 1:T
        locs = double(locationTraj(:, t));
        nets = double(networkTraj(:, t));
        M_history(:, t) = accumarray(locs, 1, [N, 1]) / Nagents;
        MIN_history(:, t) = accumarray(locs(nets == 1), 1, [N, 1]) / Nagents;
    end

    % ------------------------------------------------------------------
    % (8) Package individual trajectories and flow logs for the caller.
    % ------------------------------------------------------------------
    agentData = struct();
    agentData.location = locationTraj;
    agentData.wealth   = wealthTraj;
    agentData.state    = stateTraj;
    agentData.network  = networkTraj;
    agentData.skill    = skillTraj;

    flowLog = struct();
    flowLog.helpUsed        = helpUsedMask;
    flowLog.helpUsedProgram = programHelpMask;

    % ------------------------------------------------------------------
    % (9) Assemble the aid accounting structure with summary statistics.
    % ------------------------------------------------------------------
    aidLog = struct();
    aidLog.config = struct('name', programName, ...
                           'totalBudget', totalBudget, ...
                           'startPeriod', startPeriod, ...
                           'virtualMass', virtualMass, ...
                           'shuffleAgents', shuffleAgents);
    aidLog.totalBudget      = totalBudget;
    aidLog.totalSpent       = cumulativeSpent;
    aidLog.budgetRemaining  = max(0, totalBudget - cumulativeSpent);
    if totalBudget > 0
        aidLog.spentShare = cumulativeSpent / totalBudget;
    else
        aidLog.spentShare = 0;
    end
    aidLog.perPeriodSpent      = perPeriodSpent;
    aidLog.perPeriodRecipients = perPeriodRecipients;
    aidLog.cumulativeSpent     = cumsum(perPeriodSpent);
    aidLog.budgetPath          = budgetPath;
    aidLog.programHelpMask     = programHelpMask;
    aidLog.numRecipients       = sum(perPeriodRecipients);
end