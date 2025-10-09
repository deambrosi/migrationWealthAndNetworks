function out = SimulatedMoments(x, opt)
% SIMULATEDMOMENTS  End-to-end pipeline: params(x) -> equilibrium -> simulation -> moments.
%
%   [MVEC, OUT] = SIMULATEDMOMENTS(X, OPT) builds parameters from X,
%   solves the dynamic equilibrium, simulates agents, and computes the
%   simulated moments specified in the project notes.
%
%   INPUTS
%   ------
%   x   : vector or struct of parameter overrides (see SetParameters(dims,x)).
%   opt : optional struct to alter runtime settings (all fields optional)
%         .fast            (bool) reduce T and Nagents for speed.
%         .seed            (int)  rng seed for reproducibility.
%         .Ti              (int)  horizon index for tenure-cohort moments.
%         .max_tenure      (int)  cap for tenure bins.
%         .Nagents         (int)  override number of simulated agents.
%         .quarters/.years (int)  override total horizon (T = quarters or 4×years).
%         .burn            (int)  custom burn-in periods removed from stats.
%         .TransitionPeriods (int) quarters over which A(1)/B(1) transition.
%         .A1_start/.A1_end/.A1Transition (scalars/vector) override productivity path.
%         .B1_start/.B1_end/.B1Transition (scalars/vector) override amenity path.
%
%   OUTPUTS
%   -------
%   mvec : column vector of stacked simulated moments (see OUT.map for index ranges).
%   out  : struct with detailed outputs for diagnostics:
%          .dims, .params, .settings
%          .vf_nh, .pol_eqm
%          .M_eqm, .G_dist
%          .M_total, .M_network
%          .agentData, .flowLog
%          .moments  (struct with shaped arrays)
%          .map      (struct with indices into mvec)
%
%   NOTES
%   -----
%   • This function does not require observed data; it only simulates.
%   • Use OUT.map to align simulated moments to data moments in your GMM code.
% -------------------------------------------------------------------------

    if nargin < 1 || isempty(x)
        x = struct();
    end
    if nargin < 2 || isempty(opt)
        opt = struct();
    end

    %% 1) Initialize dimensions and settings ---------------------------------
    dims     = setDimensionParam();
    settings.it      = 0;   % Iteration counter
    settings.diffV   = 1;   % Initial V-function difference

     %Convergence tolerances
    settings.tolV    = 0.5;     % Value function convergence threshold
    settings.tolM    = 1e-2;    % Migration convergence threshold

    % Iteration limits
    settings.MaxItV  = 40;      % Maximum iterations for value function iteration
    settings.MaxItJ  = 10;      % Maximum iterations for policy update (inner loop)
    settings.MaxIter = 100;     % Maximum iterations for outer algorithm

    % Simulation configuration
    settings.Nagents = 5000;    % Number of simulated agents
    settings.T       = 100;     % Total simulated time periods
    settings.burn    = 50;      % Burn-in periods removed from statistics

    if isfield(opt, 'fast') && opt.fast
        settings.Nagents = 1000;
        settings.T       = 40;
        settings.burn    = min(20, settings.T - 1);
        settings.fastFlag = true;
    else
        settings.fastFlag = false;
    end

    if isfield(opt, 'Nagents') && ~isempty(opt.Nagents)
        settings.Nagents = max(1, round(opt.Nagents));
    end
    if isfield(opt, 'quarters') && ~isempty(opt.quarters)
        settings.T = max(1, round(opt.quarters));
    elseif isfield(opt, 'years') && ~isempty(opt.years)
        settings.T = max(1, 4 * round(opt.years));
    end
    if isfield(opt, 'burn') && ~isempty(opt.burn)
        settings.burn = max(0, min(round(opt.burn), settings.T - 1));
    else
        settings.burn = min(settings.burn, max(settings.T - 1, 0));
    end

    if isfield(opt, 'seed')
        rng(opt.seed);
        settings.seed = opt.seed;
    end

    if isfield(opt, 'Ti')
        settings.Ti = min(max(1, opt.Ti), settings.T);
    else
        settings.Ti = settings.T;
    end
    if isfield(opt, 'max_tenure')
        settings.max_tenure = opt.max_tenure;
    else
        settings.max_tenure = settings.Ti;
    end

    %% 2) Parameters, grids, and matrices ------------------------------------
    params            = SetParameters(dims, x);
    params            = applyABTransitionOverrides(params, opt, settings);
    if isfield(params, 'TransitionPeriods')
        settings.TransitionPeriods = params.TransitionPeriods;
    end
    [grids, indexes]  = setGridsAndIndices(dims);
    matrices          = constructMatrix(dims, params, grids, indexes);
    params.P          = matrices.P;    % convenience handle for downstream code

    %% 3) Steady-state (no-help) value functions -----------------------------
    [vf_nh, ~] = noHelpEqm(dims, params, grids, indexes, matrices, settings);

    %% 4) Initial distributions and dynamic equilibrium ----------------------
    m0 = createInitialDistribution(dims, settings);

    M0          = zeros(dims.N, settings.T);
    M0(1, :)    = 1;

    [pol_eqm, M_eqm] = solveDynamicEquilibrium(M0, vf_nh, m0, ...
        dims, params, grids, indexes, matrices, settings); 

    %% 5) Help path and simulation -------------------------------------------
    G_dist = computeG(M_eqm, params.ggamma);                  % [H×T]
    [M_total, M_network, agentData, flowLog] = simulateAgents( ...
        m0, pol_eqm, G_dist, dims, params, grids, matrices, settings);

    %% 6) Compute simulated moments -----------------------------------------
    if ~isempty(flowLog)
        agentData.flowLog = flowLog;  % attach for moment construction
    end
    moments = computeSimulatedMoments(agentData, M_total, M_network, ...
        dims, params, grids, settings, matrices);
    if isfield(agentData, 'flowLog')
        agentData = rmfield(agentData, 'flowLog');
    end

    %[mvec, map] = packMoments(moments, dims, settings);

    %% 7) Pack outputs -------------------------------------------------------
    out.dims      = dims;
    out.params    = params;
    out.settings  = settings;
    out.vf_nh     = vf_nh;
    out.pol_eqm   = pol_eqm;
    out.M_eqm     = M_eqm;
    out.G_dist    = G_dist;
    out.M_total   = M_total;
    out.M_network = M_network;
    out.agentData = agentData;
    out.flowLog   = flowLog;
    out.moments   = moments;
    %out.map       = map;
    out.matrices  = matrices;

end

%% Local helpers --------------------------------------------------------------
function params = applyABTransitionOverrides(params, opt, settings)
% APPLYABTRANSITIONOVERRIDES  Inject time paths for A(1) and B(1) when requested.
%
%   PARAMS = APPLYABTRANSITIONOVERRIDES(PARAMS, OPT, SETTINGS) mirrors the
%   transition logic used in FASTSIMULATEDMOMENTS, allowing the dynamic
%   equilibrium solver to handle temporary movements in fundamentals. The
%   routine updates params.A(1) and params.B(1) to their terminal values and
%   stores full time paths in params.A_timePath / params.B_timePath.

    if settings.T <= 0
        return;
    end

    transitionDefault = 12;
    if isfield(opt, 'TransitionPeriods') && ~isempty(opt.TransitionPeriods)
        requestedTransitionPeriods = max(1, round(opt.TransitionPeriods));
    else
        requestedTransitionPeriods = transitionDefault;
    end

    transitionPeriods = max(1, min(requestedTransitionPeriods, max(settings.T - 1, 1)));

    A1_start = params.A(1);
    A1_end   = params.A(1);
    if isfield(opt, 'A1Transition') && numel(opt.A1Transition) >= 2
        A1_start = opt.A1Transition(1);
        A1_end   = opt.A1Transition(end);
    else
        if isfield(opt, 'A1_start') && ~isempty(opt.A1_start)
            A1_start = opt.A1_start;
        end
        if isfield(opt, 'A1_end') && ~isempty(opt.A1_end)
            A1_end = opt.A1_end;
        end
    end

    B1_start = params.B(1);
    B1_end   = params.B(1);
    if isfield(opt, 'B1Transition') && numel(opt.B1Transition) >= 2
        B1_start = opt.B1Transition(1);
        B1_end   = opt.B1Transition(end);
    else
        if isfield(opt, 'B1_start') && ~isempty(opt.B1_start)
            B1_start = opt.B1_start;
        end
        if isfield(opt, 'B1_end') && ~isempty(opt.B1_end)
            B1_end = opt.B1_end;
        end
    end

    if transitionPeriods == 1
        A_schedule = A1_start;
        B_schedule = B1_start;
    else
        A_schedule = linspace(A1_start, A1_end, transitionPeriods);
        B_schedule = linspace(B1_start, B1_end, transitionPeriods);
    end

    params.A(1) = A1_end;
    params.B(1) = B1_end;

    params.TransitionPeriods = transitionPeriods;
    params.A1_schedule       = A_schedule(:);
    params.B1_schedule       = B_schedule(:);
    params.A1_transition     = [A1_start; A1_end];
    params.B1_transition     = [B1_start; B1_end];

    T = max(settings.T, 1);
    params.A_timePath = repmat(params.A(:), 1, T);
    params.B_timePath = repmat(params.B(:), 1, T);

    horizonForPath = min(T, transitionPeriods);
    for tt = 1:horizonForPath
        params.A_timePath(:, tt) = params.A(:);
        params.B_timePath(:, tt) = params.B(:);
        params.A_timePath(1, tt) = A_schedule(min(tt, numel(A_schedule)));
        params.B_timePath(1, tt) = B_schedule(min(tt, numel(B_schedule)));
    end
end
