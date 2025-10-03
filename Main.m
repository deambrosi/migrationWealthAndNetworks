%% Main Script: Solve the Steady-State and Transition Dynamics of the Model
% This script performs the following:
%   1) Initializes model parameters, grids, and functional forms.
%   2) Solves for the steady-state value and policy functions assuming no help (G = G0).
%   3) Solves the dynamic equilibrium through backward induction and forward simulation.
%   4) Simulates final trajectories using the converged dynamic policy.
%
% AUTHOR : Agustin Deambrosi
% DATE   : September 2025
% VERSION: 2.1
% ==============================================================================

clc; clear; close all;

%% 1. Initialize Model Parameters, Grids, and Settings
fprintf('Initializing model parameters and grids...\n');
try
    dims                = setDimensionParam();                            % Dimensions (S, N, k, K, H, Na, na)
    params              = SetParameters(dims);                            % Structural parameters
    [grids, indexes]    = setGridsAndIndices(dims);                       % Grids and index matrices
    matrices            = constructMatrix(dims, params, grids, indexes);  % Precomputed utility, P, and τ-eff
    settings            = IterationSettings();                            % Iteration controls and simulation length
    m0                  = createInitialDistribution(dims, settings);      % Initial agent distribution

    % (optional) keep P in params for functions that expect it
    params.P            = matrices.P;

    fprintf('Initialization completed successfully.\n');
catch ME
    error('Error during initialization: %s', ME.message);
end

tic;

%% 2. Solve No-Help Equilibrium (Steady-State)
fprintf('\nComputing No-Help Value and Policy Functions...\n');
try
    [vf_nh, pol_nh] = noHelpEqm(dims, params, grids, indexes, matrices, settings);  % Solve with G = G0
    fprintf('No-Help Value and Policy Functions found successfully.\n');
catch ME
    error('Error computing No-Help equilibrium: %s', ME.message);
end

%% 3. Initialize Guess for Dynamic Network Agent Distribution
% We assume all agents begin in location 1 and are part of the network.
M_init          = zeros(dims.N, 1);
M_init(1)       = 1;
M0              = repmat(M_init, 1, settings.T);  % [N x T] initial guess for network masses of n=1

%% 4. Solve Dynamic Equilibrium (Transition Path)
fprintf('\nSolving Dynamic Equilibrium via Backward Induction and Simulation...\n');
try
    [pol_eqm, M_eqm, it_count] = solveDynamicEquilibrium(M0, vf_nh, m0, ...
        dims, params, grids, indexes, matrices, settings);
    fprintf('Dynamic equilibrium solved successfully in %d iterations.\n', it_count);
catch ME
    error('Error solving dynamic equilibrium: %s', ME.message);
end

%% 5. Simulate Final Trajectories Using Converged Policy
fprintf('\nSimulating Agent Paths Using Converged Policies...\n');
try
    % (A) Compute the full time path of help-PMFs in one call (H x T)
    %     G_t(:,t) = g(h | M_eqm(:,t)) with independence across destinations
    G_dist = computeG(M_eqm, params.ggamma);   % [dims.H x settings.T]

    % (B) Forward simulation under converged policy and endogenous help environment
    %     NOTE: simulateAgents signature here expects 'matrices' (for P and τ-eff)
    [M_total, M_network, agentData] = simulateAgents( ...
        m0, pol_eqm, G_dist, dims, params, grids, matrices, settings);

    % (C) Simple diagnostics (optional but useful)
    colsum_total   = sum(M_total,   1);    % should be ~1 each period
    colsum_network = sum(M_network, 1);    % should be <= 1 each period

    if any(abs(colsum_total - 1) > 1e-10)
        warning('Column sums of M_total are not ~1 in some periods. Max dev: %.3e', max(abs(colsum_total - 1)));
    end
    if any(colsum_network - 1 > 1e-10)
        warning('Column sums of M_network exceed 1 in some periods. Max excess: %.3e', max(colsum_network - 1));
    end

    fprintf('Simulation completed successfully.\n');
catch ME
    error('Error during final simulation: %s', ME.message);
end

%% 6. (Optional) Save Outputs / Timing
% save('results_main.mat', 'dims', 'params', 'grids', 'indexes', 'matrices', ...
%      'vf_nh', 'pol_nh', 'pol_eqm', 'M_eqm', 'M_total', 'M_network', 'agentData');

elapsed = toc;
fprintf('\nTotal runtime: %.2f seconds.\n', elapsed);
