function Ue = computeUtilityMatrix(dims, params, grids, indexes)
% COMPUTEUTILITYMATRIX  Recompute period utility matrix for given parameters.
%
%   Ue = COMPUTEUTILITYMATRIX(DIMS, PARAMS, GRIDS, INDEXES) returns the
%   per-period utility evaluated on the fine asset grid, consistent with the
%   objects produced by CONSTRUCTMATRIX. This helper isolates the portion of
%   CONSTRUCTMATRIX that depends on location-specific productivity (A) and
%   amenities (B), allowing callers to refresh MATRICES.Ue when these
%   parameters vary over time.
%
%   INPUTS
%   ------
%   dims    : struct with model dimensions (S, K, Na, N, etc.).
%   params  : struct of structural parameters (A, B, theta_s, theta_k, xi, ...).
%   grids   : struct containing the asset and shock grids.
%   indexes : struct with reshaped index helpers (see setGridsAndIndices).
%
%   OUTPUT
%   ------
%   Ue      : array [S×K×1×N×na] with period utility on the fine asset grid.
%
%   NOTES
%   -----
%   • The implementation mirrors the utility block in CONSTRUCTMATRIX to keep
%     both routines in sync.
%   • Infeasible consumption (≤0) is penalized with −realmax to preserve the
%     original maximization logic.
%
%   See also CONSTRUCTMATRIX.
%
%   AUTHOR: OpenAI Assistant (adapting existing project code)
%   DATE  : 2025-02-14
% -------------------------------------------------------------------------

    idx_SN   = sub2ind([dims.S, dims.N], indexes.I_sp, indexes.I_Np);
    theta_sn = params.theta_s(idx_SN);

    income = (2 - indexes.I_ep) .* params.bbi(indexes.I_Np) + ...
             (indexes.I_ep - 1) .* params.A(indexes.I_Np) .* ...
             theta_sn .* (1 + grids.psi(indexes.I_psip)).^(params.theta_k);

    cons = (1 / params.bbeta) .* grids.agrid(indexes.I_ap) + ...
           income - grids.ahgrid(indexes.I_app);

    amenity_weight = params.B(indexes.I_Np) .* ...
                     (1 + params.xi .* grids.psi(indexes.I_psip)).^(params.phiH);

    Ue          = zeros(size(cons));
    feasible    = cons > 0;
    Ue(feasible)  = amenity_weight(feasible) .* cons(feasible).^(0.5);
    Ue(~feasible) = -realmax;
end
