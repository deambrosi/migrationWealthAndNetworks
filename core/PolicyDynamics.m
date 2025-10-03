function [vf_path, pol_path] = PolicyDynamics(M1, vf_terminal, dims, params, grids, indexes, matrices, settings)
% POLICYDYNAMICS  Backward induction of value/policy functions along a path of M_t.
%
%   [VF_PATH, POL_PATH] = POLICYDYNAMICS(M1, VF_TERMINAL, DIMS, PARAMS, GRIDS, INDEXES, MATRICES, SETTINGS)
%   computes time-varying value and policy functions along a guessed dynamic
%   path of network masses M1(:,t), t = 1..T. It uses:
%       • backward induction for value/policy from t = T-1 down to 1
%       • time-varying help distributions G_t = g(h | M1(:,t))
%
%   INPUTS
%   ------
%   M1          : [N × T] path of network agent mass by location and time.
%   vf_terminal : struct with steady-state value functions at T (boundary):
%                   .V, .Vn, .R, .Rn     (all [S × K × Na × N])
%   dims        : struct with model dimensions (S, K, Na, N, H, …)
%   params      : struct with parameters (includes .ggamma)
%   grids       : struct with grids (agrid, ahgrid, …)
%   indexes     : struct with indexing helpers (see setGridsAndIndices)
%   matrices    : struct with precomputed matrices (Ue, a_prime, P, …)
%   settings    : struct with simulation/iteration controls (T, etc.)
%
%   OUTPUTS
%   -------
%   vf_path     : T×1 cell array of value-function structs.
%                 vf_path{t} has fields .V, .Vn, .R, .Rn for time t.
%                 By construction, vf_path{T} = vf_terminal (boundary).
%
%   pol_path    : struct of (T-1)×1 cell arrays with policy functions:
%                   .a{t}   [S×K×Na×N]      asset argmax (no network)
%                   .an{t}  [S×K×Na×N]      asset argmax (networked)
%                   .mu{t}  [S×K×Na×N×N]    migration probs (no network)
%                   .mun{t} [S×K×Na×N×N×H]  migration probs (networked)
%
%   NOTES
%   -----
%   • The boundary condition at T pins down continuation values at t = T-1.
%   • Help probabilities are computed as G_t = computeG(M1(:,t), γ).
%   • All shapes are consistent with updateValueAndPolicy.m and simulateAgents.m.
%
%   AUTHOR: Agustín Deambrosi
%   LAST REVISED: September 2025
% ======================================================================

    %% 1) Setup
    T = settings.T;

    % Preallocate outputs
    vf_path        = cell(T, 1);               % store values for t = 1..T
    pol_path.a     = cell(T-1, 1);             % decisions for t = 1..T-1
    pol_path.an    = cell(T-1, 1);
    pol_path.mu    = cell(T-1, 1);
    pol_path.mun   = cell(T-1, 1);

    % Time-varying help distributions along M1
    %   G_path: [H × T], where G_path(:,t) = g(h | M1(:,t))
    G_path = computeG(M1, params.ggamma);

    % Terminal boundary condition
    vf_path{T} = vf_terminal;

    % Productivity path (N×T). Ensure availability for backward induction.
    if ~isfield(params, 'A_path')
        error('PolicyDynamics:MissingAPath', ...
            'params.A_path is required for time-varying productivity.');
    elseif size(params.A_path, 2) < T
        error('PolicyDynamics:InvalidAPath', ...
            'params.A_path must have at least T=%d columns.', T);
    end
    A_path = params.A_path;

    %% 2) Backward induction: t = T-1 down to 1
    for t = T-1:-1:1
        % Help PMF for this period
        G_t = G_path(:, t);    % [H × 1]

        % Compute time-t value and policy given continuation at t+1
        A_t = A_path(:, t);

        [vf_t, pol_t] = updateValueAndPolicy( ...
            vf_path{t+1}, dims, params, grids, indexes, matrices, G_t, A_t);

        % Store results
        vf_path{t}   = vf_t;
        pol_path.a{t}   = pol_t.a;
        pol_path.an{t}  = pol_t.an;
        pol_path.mu{t}  = pol_t.mu;
        pol_path.mun{t} = pol_t.mun;
    end
end
