function [pol_eqm, M_eqm, it_count] = solveDynamicEquilibrium(M0, vf_terminal, m0, dims, params, grids, indexes, matrices, settings)
% SOLVEDYNAMICEQUILIBRIUM  Solve the dynamic migration equilibrium.
%
%   [POL_EQM, M_EQM, IT_COUNT] = SOLVEDYNAMICEQUILIBRIUM(M0, VF_TERMINAL, M0, DIMS, PARAMS, GRIDS, INDEXES, MATRICES, SETTINGS)
%   iterates on the time path of network masses M_t using:
%     • Backward induction (POLICYDYNAMICS) to compute time‑varying policies
%       given a guessed path of M_t and the terminal boundary condition,
%     • Forward simulation (SIMULATEAGENTS) to update the path of M_t,
%     • A weighted convergence metric emphasizing early periods.
%
%   INPUTS
%   ------
%   M0          : [N × T] initial guess for network agent shares by location over time.
%   vf_terminal : struct with steady‑state/terminal value functions at T (.V, .Vn, .R, .Rn).
%   m0          : [Nagents × 1] array of agent structs (initial conditions).
%   dims        : struct with model dimensions (S, K, Na, N, H, …).
%   params      : struct with parameters (.ggamma, .cchi, etc.).
%   grids       : struct with grids (.agrid, .ahgrid, …).
%   indexes     : struct with indexing helpers.
%   matrices    : struct with precomputed matrices (.Ue, .a_prime, .P, …).
%   settings    : struct with iteration/simulation controls (.T, .tolM, .MaxItJ, …).
%
%   OUTPUTS
%   -------
%   pol_eqm  : struct of (T−1)×1 cell arrays with policy functions
%                .a{t}   [S×K×Na×N]      (no network, assets)
%                .an{t}  [S×K×Na×N]      (network, assets)
%                .mu{t}  [S×K×Na×N×N]    (no network, migration)
%                .mun{t} [S×K×Na×N×N×H]  (network, migration by help h)
%   M_eqm    : [N × T] converged path of network agent shares by location.
%   it_count : scalar, number of outer iterations to convergence (or limit).
%
%   NOTES
%   -----
%   • The help PMF g(h|M_t) is recomputed each outer iteration from the current guess M_eqm
%     and used consistently in BOTH backward induction and forward simulation.
%   • Convergence metric: weighted L1 difference by period:
%       diff_per_t = sum_i |M_old(i,t) − M_new(i,t)| / (1 + |M_new(i,t)|)
%       diffM = sum_t w_t * diff_per_t, with w_t decaying geometrically.
%
%   AUTHOR: Agustín Deambrosi
%   LAST REVISED: September 2025
% ======================================================================

    %% 1) Initialization ---------------------------------------------------
    T        = settings.T;
    M_eqm    = M0;         % current guess for M path (N×T)
    diffM    = 1.0;        % convergence metric
    it_count = 0;          % iteration counter

    % Exponential time weights: emphasize early periods (t=1 highest)
    beta_weight  = 0.7;                 % tweak if needed
    time_weights = beta_weight .^ (0:T-1);
    time_weights = time_weights / sum(time_weights);   % normalize to 1

    fprintf('\nSolving Dynamic Equilibrium...\n');

    %% 2) Fixed‑point iteration on M_t ------------------------------------
    while (diffM > settings.tolM) && (it_count < settings.MaxItJ)
        it_count = it_count + 1;

        % (a) Backward induction of time‑varying policies at this M path
        [~, pol_new] = PolicyDynamics(M_eqm, vf_terminal, ...
                                      dims, params, grids, indexes, matrices, settings);

        % (b) Forward simulation: use time‑varying help PMFs implied by M_eqm
        G_path = computeG(M_eqm, params.ggamma);          % [H×T]
        [~, M_new, ~] = simulateAgents(m0, pol_new, G_path, ...
                                       dims, params, grids, matrices, settings);

        % (c) Weighted convergence metric over time
        diff_per_t = sum(abs(M_eqm - M_new) ./ (1 + abs(M_new)), 1);  % 1×T
        diffM      = sum(time_weights .* diff_per_t);                 % scalar

        % (d) Update the path (optionally add relaxation here if needed)
        M_eqm = M_new;

        % (e) Progress report
        fprintf('  Iteration %d: weighted diffM = %.6e\n', it_count, diffM);
    end

    % Final policy path from the last iteration
    pol_eqm = pol_new;
end
