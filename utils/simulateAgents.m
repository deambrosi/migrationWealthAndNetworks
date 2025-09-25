function [M_history, MIN_history, agentData, flowLog] = simulateAgents(m0, pol, G_dist, dims, params, grids, matrices, settings)
% SIMULATEAGENTS  Simulate agent evolution over T periods using policy paths.
%
%   [M_HISTORY, MIN_HISTORY, AGENTDATA] = SIMULATEAGENTS(M0, POL, G_DIST, ...)
%   simulates the dynamic paths of agents given time-varying policy functions
%   and help-offer distributions. It returns total location shares, networked
%   shares, and individual trajectories. When a fourth output is requested, the
%   function also logs migration-flow diagnostics (help usage and direct-from-
%   origin moves) without affecting existing calls.
%   INPUTS
%   ------
%   m0      : [Nagents × 1] struct array of initial agent states with fields:
%               .skill    (1..S)
%               .state    (1..K)   — packed (z,ψ): rows 1..k=U(ψ=1..k), rows k+1..2k=E(ψ=1..k)
%               .wealth   (1..Na)
%               .location (1..N)
%               .network  (0/1)
%
%   pol     : struct of (T-1)×1 cell arrays (time-varying policies)
%               .a{t}   [S×K×Na×N]           asset argmax (no network)
%               .an{t}  [S×K×Na×N]           asset argmax (networked)
%               .mu{t}  [S×K×Na×N×N]         migration probs (no network)
%               .mun{t} [S×K×Na×N×N×H]       migration probs (networked, indexed by help h)
%            (Time-invariant case allowed: numeric arrays will be replicated to cells.)
%
%   G_dist  : [H × T] time-varying help PMFs; column t is g(h | M_t)
%
%   dims    : struct with dimensions (S, K, Na, N, H, T, …)
%
%   params  : struct with parameters (may include .cchi; transitions .P are read from MATRICES)
%
%   grids   : struct with grids
%               .agrid  [Na×1]   coarse asset grid
%               .ahgrid [na×1]   fine asset grid
%
%   matrices: struct with precomputed arrays
%               .P          [S×K×K×N]  (z,ψ) transition when staying in location i
%               .mig_costs  [N×N×H]    effective migration costs τ^{iℓ}(h)
%
%   settings: struct with simulation settings (.Nagents, .T)
%
%   OUTPUTS
%   -------
%   M_history   : [N × T]   total mass per location per period (shares sum to 1)
%   MIN_history : [N × T]   mass of networked agents per location per period
%   agentData   : struct with trajectories
%                   .location [Nagents×T]
%                   .wealth   [Nagents×T]
%                   .state    [Nagents×T]
%                   .network  [Nagents×T]
%                   .skill    [Nagents×1]
%   flowLog     : (optional) struct with migration-flow flags (only if requested)
%                   .helpUsed       [Nagents×T] logical, true when moved with help
%                   .directFromVzla [Nagents×T] logical, true when moved from origin
%
%   NOTES
%   -----
%   • Decisions are applied for periods t = 1..T-1; period T only records states.
%   • Migration resets to K=1 (U, ψ₁) by construction.
%   • If any PMF column G_dist(:,t) sums ≈ 0, it is renormalized defensively.
%
%   AUTHOR: Agustín Deambrosi (revised)
%   LAST REVISED: September 2025
% ======================================================================

    %% 1) Setup ------------------------------------------------------------
    T         = settings.T;
    Nagents   = settings.Nagents;
    N         = dims.N;
    logFlows  = nargout >= 4;


    % Accept either time-varying (cell) or fixed (numeric) policies
    isTimeInvariant = ~iscell(pol.a);
    if isTimeInvariant
        pol.a   = repmat({pol.a},  T-1, 1);
        pol.an  = repmat({pol.an}, T-1, 1);
        pol.mu  = repmat({pol.mu}, T-1, 1);
        pol.mun = repmat({pol.mun},T-1, 1);
    end

    % (z,ψ) transition and effective migration costs (precomputed in MATRICES)

    P_local   = matrices.P;          % [S×K×K×N]
    mig_costs = matrices.mig_costs;  % [N×N×H]
    if logFlows && isfield(matrices, 'Hbin')
        Hbin = matrices.Hbin;        % [H×N]
    else
        Hbin = [];
    end

    %% 2) Preallocate trajectories ----------------------------------------
    locationTraj = zeros(Nagents, T, 'uint16');
    wealthTraj   = zeros(Nagents, T, 'uint16');
    stateTraj    = zeros(Nagents, T, 'uint16');   % K-index
    networkTraj  = zeros(Nagents, T, 'uint8');
    skillTraj    = zeros(Nagents, 1, 'uint16');   % skill fixed
    if logFlows
        helpUsedTraj   = false(Nagents, T);
        directVzlaTraj = false(Nagents, T);
    end


    %% 3) Simulation loop --------------------------------------------------
    parfor agentIdx = 1:Nagents
        % --- Initialize agent ---
        agent = m0(agentIdx);

        loc = uint16(agent.location);
        wea = uint16(agent.wealth);
        sta = uint16(agent.state);
        net = uint8(agent.network);
        ski = uint16(agent.skill);

        % Trajectory buffers
        locHist = zeros(1, T, 'uint16');
        weaHist = zeros(1, T, 'uint16');
        staHist = zeros(1, T, 'uint16');
        netHist = zeros(1, T, 'uint8');
        helpHist   = false(1, T);
        directHist = false(1, T);


        locHist(1) = loc;
        weaHist(1) = wea;
        staHist(1) = sta;
        netHist(1) = net;

        for t = 1:(T-1)

            %% A) Asset decision (choose fine-grid index, then map to coarse index)
            if net == 1
                a_fine = pol.an{t}(ski, sta, wea, loc);
            else
                a_fine = pol.a{t}(ski, sta, wea, loc);
            end
            % Clamp to [1, length(ahgrid)]
            a_fine = uint32(min(max(a_fine, 1), length(grids.ahgrid)));

            % Map chosen fine asset (ahgrid) back to nearest coarse index on agrid
            [~, nextWea] = min(abs(grids.agrid - grids.ahgrid(a_fine)));
            nextWea = uint16(max(1, min(nextWea, numel(grids.agrid))));

            %% B) Migration decision (draw help if networked, then draw destination)
            if net == 1
                % Draw help-vector index according to current PMF G_dist(:,t)
                G_t   = G_dist(:, t);
                G_t   = G_t / max(sum(G_t), eps);          % normalize defensively
                h_idx = uint16(find(cumsum(G_t) >= rand(), 1));
                if isempty(h_idx), h_idx = uint16(1); end

                % Migration probs over destinations (size N)
                migProb = squeeze(pol.mun{t}(ski, sta, nextWea, loc, :, h_idx));
            else
                h_idx  = uint16(1);                         % default "no-help" slice
                migProb = squeeze(pol.mu{t}(ski, sta, nextWea, loc, :));
            end

            % Normalize and guard against degeneracy
            sProb = sum(migProb);
            if sProb <= 0 || ~isfinite(sProb)
                migProb = zeros(N,1); migProb(loc) = 1;     % fallback: stay
            else
                migProb = migProb ./ sProb;
            end

            % Draw next location
            cdf     = cumsum(migProb);
            r       = rand();
            nextLoc = uint16(find(cdf >= r, 1));
            if isempty(nextLoc)
                [~, argmax] = max(migProb);
                nextLoc = uint16(argmax);
            end

            %% C) Wealth and K-state transition
            moved = nextLoc ~= loc;

            if logFlows
                helpFlag   = false;
                directFlag = false;
                if moved
                    helpFlag   = (net == 1) && ~isempty(Hbin) && Hbin(double(h_idx), double(nextLoc)) == 1;
                    directFlag = (loc == 1);
                end
                helpHist(t+1)   = helpFlag;
                directHist(t+1) = directFlag;
            end

            if moved
                % Pay migration cost (depends on help vector)
                migCost = mig_costs(loc, nextLoc, h_idx);
                newA    = grids.agrid(nextWea) - migCost;

                % Map back to nearest coarse-grid index and clamp
                [~, nextWea] = min(abs(grids.agrid - newA));
                nextWea = uint16(max(1, min(nextWea, numel(grids.agrid))));

                % Reset (z,ψ) → (U, ψ₁), which is packed K-index 1 by construction
                sta = uint16(1);

            else
                % Stay: draw next K via P(s,:,: ,loc); rows=current K, cols=next K
                P_row  = squeeze(P_local(ski, sta, :, loc));   % 1×K
                rowSum = sum(P_row);
                if rowSum > 0 && isfinite(rowSum)
                    P_row = P_row ./ rowSum;
                    transCdf = cumsum(P_row);
                    sta_next = uint16(find(transCdf >= rand(), 1));
                    if isempty(sta_next), sta_next = sta; end
                    sta = sta_next;
                else
                    % Fallback: remain in same K (rare)
                    sta = sta;
                end
            end

            %% D) Network status update (erosion outside origin)
            if net == 1 && nextLoc ~= 1
                chi = params.cchi;
                if isscalar(chi)
                    pLose = chi;
                elseif isequal(size(chi), [dims.S, dims.N])
                    pLose = chi(ski, nextLoc);
                elseif isequal(size(chi), [1, dims.N])
                    pLose = chi(nextLoc);
                else
                    pLose = chi; % best effort if malformed
                end
                if rand() < pLose
                    net = uint8(0);
                end
            end

            %% E) Save next-period state
            loc = nextLoc;
            wea = nextWea;

            locHist(t+1) = loc;
            weaHist(t+1) = wea;
            staHist(t+1) = sta;
            netHist(t+1) = net;
        end

        % Commit trajectories for this agent
        locationTraj(agentIdx, :) = locHist;
        wealthTraj( agentIdx, :)  = weaHist;
        stateTraj(  agentIdx, :)  = staHist;
        networkTraj(agentIdx, :)  = netHist;
        skillTraj(  agentIdx, 1)  = ski;
        if logFlows
            helpUsedTraj(agentIdx, :)   = helpHist;
            directVzlaTraj(agentIdx, :) = directHist;
        end
    end

    %% 4) Aggregate location histories -----------------------------------
    M_history   = zeros(N, T);
    MIN_history = zeros(N, T);

    for t = 1:T
        locs = double(locationTraj(:, t));
        nets = double(networkTraj(:, t));

        M_history(:, t)   = accumarray(locs, 1, [N, 1]) / Nagents;
        MIN_history(:, t) = accumarray(locs(nets == 1), 1, [N, 1]) / Nagents;
    end

    %% 5) Output trajectories ---------------------------------------------
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

