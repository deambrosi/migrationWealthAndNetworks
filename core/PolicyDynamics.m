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
%   • When params contains time paths (e.g., params.A_timePath), the routine
%     refreshes MATRICES.Ue period-by-period so that value updates reflect the
%     current productivity/amenity levels.
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

    % Flags for time-varying fundamentals
    hasApath = isfield(params, 'A_timePath') && ~isempty(params.A_timePath);
    hasBpath = isfield(params, 'B_timePath') && ~isempty(params.B_timePath);
    if hasApath
        rowsA = min(size(params.A_timePath, 1), numel(params.A));
        colsA = size(params.A_timePath, 2);
    else
        rowsA = 0; colsA = 0; %#ok<NASGU>
    end
    if hasBpath
        rowsB = min(size(params.B_timePath, 1), numel(params.B));
        colsB = size(params.B_timePath, 2);
    else
        rowsB = 0; colsB = 0; %#ok<NASGU>
    end

    %% 2) Backward induction: t = T-1 down to 1
    for t = T-1:-1:1
        % Help PMF for this period
        G_t = G_path(:, t);    % [H × 1]

        % Compute time-t value and policy given continuation at t+1
        if hasApath || hasBpath
            params_t = params;
            matrices_t = matrices;

            if hasApath
                colA = min(t, colsA);
                params_t.A(1:rowsA) = params.A_timePath(1:rowsA, colA);
            end
            if hasBpath
                colB = min(t, colsB);
                params_t.B(1:rowsB) = params.B_timePath(1:rowsB, colB);
            end

            matrices_t.Ue = computeUtilityMatrix(dims, params_t, grids, indexes);
        else
            params_t   = params;
            matrices_t = matrices;
        end

        [vf_t, pol_t] = updateValueAndPolicy( ...
            vf_path{t+1}, dims, params_t, grids, indexes, matrices_t, G_t);

        % Store results
        vf_path{t}   = vf_t;
        pol_path.a{t}   = pol_t.a;
        pol_path.an{t}  = pol_t.an;
        pol_path.mu{t}  = pol_t.mu;
        pol_path.mun{t} = pol_t.mun;
    end
end
