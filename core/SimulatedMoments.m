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
%         .fast       (bool) if true, reduce T and Nagents for speed.
%         .seed       (int)  rng seed for reproducibility.
%         .Ti         (int)  horizon index for tenure-cohort moments (default: settings.T).
%         .max_tenure (int)  cap for tenure bins (default: Ti).
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
