function matrices = constructMatrix(dims, params, grids, indexes)
% CONSTRUCTMATRIX  Precompute static matrices for utility, wealth, and transitions.
%
%   MATRICES = CONSTRUCTMATRIX(DIMS, PARAMS, GRIDS, INDEXES) builds objects
%   that are constant across iterations of the value function, including
%   consumption utilities, asset grids adjusted for migration costs, and
%   the employment–integration transition operator.
%
%   INPUTS
%   ------
%   dims    : struct
%       Dimension parameters (S, N, K, Na, H, etc.).
%   params  : struct
%       Model parameters (β, A, B, θ_s, θ_k, bbi, τ, α, γ, χ, etc.).
%   grids   : struct
%       Grids for assets and integration (agrid, ahgrid, ψ).
%   indexes : struct
%       Precomputed index arrays for reshaping and logic (see setGridsAndIndices).
%
%   OUTPUT
%   ------
%   matrices : struct with fields
%       .Ue        Utility from feasible consumption (−∞ for infeasible).
%       .a_prime   Assets net of migration costs (S×K×Na×N×N×H).
%       .mig_costs Raw effective migration costs τ^{iℓ}(h) (N×N×H).
%       .Hbin      Binary help matrix enumerating h (H×N).
%       .P         Transition operator for (employment × ψ),
%                  shape S×K×K×N (built via build_P).
%
%   NOTES
%   -----
%   • The struct is designed to avoid recomputing these large matrices at
%     every iteration of the Bellman recursion.
%   • Utility is penalized with −realmax when consumption is ≤0.
%   • Migration costs account for network help via build_tau_eff.
%
%   DEPENDENCIES
%   ------------
%   • build_tau_eff.m : expands τ^{iℓ} into τ^{iℓ}(h) across help vectors.
%   • build_P.m       : builds the (z,ψ) transition matrix.
%
%   AUTHOR: Agustín Deambrosi
%   LAST REVISED: September 2025
% ======================================================================

    %% 1. Consumption and utility ------------------------------------------------
    % Linear index to pick the right θ_s for each (skill, location)
    idx_SN   = sub2ind([dims.S, dims.N], indexes.I_sp, indexes.I_Np);
    theta_sn = params.theta_s(idx_SN);

    % Income flow:
    %   • Unemployed: b_i (location-specific benefit).
    %   • Employed  : A_i * θ^s_i * (1+ψ)^(θ_k).
    income = (2 - indexes.I_ep) .* params.bbi(indexes.I_Np) + ...
             (indexes.I_ep - 1) .* params.A(indexes.I_Np) .* ...
             theta_sn .* (1 + grids.psi(indexes.I_psip)).^(params.theta_k);

    % Consumption = gross return on assets + income − savings
    cons = (1 / params.bbeta) .* grids.agrid(indexes.I_ap) + ...
           income - grids.ahgrid(indexes.I_app);

    % Amenity scaling: B_i × (1 + ξ·ψ)^(φH)
    amenity_weight = params.B(indexes.I_Np) .* ...
                     (1 + params.xi .* grids.psi(indexes.I_psip)).^(params.phiH);

    % Period utility from consumption (log utility with amenity weight)
    Ue          = zeros(size(cons));
    feasible    = cons > 0;
    Ue(feasible)  = amenity_weight(feasible) .* log(cons(feasible));
    Ue(~feasible) = -realmax;   % penalize infeasible consumption

    %% 2. After-migration wealth -------------------------------------------------
    % Effective migration cost tensor: τ^{iℓ}(h), N×N×H
    [mig_costs, Hbin] = build_tau_eff(dims, params);

    % Expand to match (S,K,Na,N,N,H) dimensions:
    %   permute to add singleton skill/asset/state dimensions,
    %   then replicate across S,K,Na.
    eff_mig_costs = permute(mig_costs, [4, 5, 6, 1, 2, 3]);
    eff_mig_costs = repmat(eff_mig_costs, dims.S, dims.K, dims.Na, 1, 1, 1);

    % Assets net of migration costs
    a_prime = grids.agrid(indexes.I_am) - eff_mig_costs;

    %% 3. Transition operator for (employment × ψ) -------------------------------
    % Build P: S×K×K×N (row-stochastic in K for each skill and location)
    matrices.P = build_P(dims, params);

    %% 4. Pack outputs -----------------------------------------------------------
    matrices.Ue        = Ue;
    matrices.a_prime   = a_prime;
    matrices.mig_costs = mig_costs;
    matrices.Hbin      = Hbin;

end
