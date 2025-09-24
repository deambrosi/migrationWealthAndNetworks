function [tau_eff, Hbin] = build_tau_eff(dims, params)
% BUILD_TAU_EFF  Construct effective migration costs with help offers.
%
%   [TAU_EFF, HBIN] = BUILD_TAU_EFF(DIMS, PARAMS) expands the base migration
%   cost matrix τ^{iℓ} into τ^{iℓ}(h) for every possible help vector h ∈ {0,1}^N.
%   When a help offer is present for destination ℓ (h^ℓ = 1), the cost is 
%   reduced by the factor α ∈ (0,1). Otherwise the base cost applies.
%
%   INPUTS
%   ------
%   dims : struct
%       .N   Number of locations
%       .H   Number of help vectors (2^N)
%
%   params : struct
%       .ttau    [N × N] base migration costs τ^{iℓ}, with τ^{ii} = 0
%       .aalpha  Scalar α in (0,1), cost reduction factor when helped
%
%   OUTPUTS
%   -------
%   tau_eff : [N × N × H] array
%       Effective migration costs τ^{iℓ}(h) for each origin i, destination ℓ,
%       and help vector index h.
%
%   Hbin    : [H × N] binary matrix
%       Enumeration of all help vectors. Row k = h(k,:) gives the help
%       indicator for each destination (1 = help, 0 = no help).
%
%   NOTES
%   -----
%   • The “help” applies only to the destination dimension ℓ.
%   • This routine is used in simulation and value-function iteration to
%     adjust feasible migration sets depending on realized help.
%
%   AUTHOR: Agustín Deambrosi
%   LAST REVISED: September 2025
% ======================================================================

    N     = dims.N;
    H     = dims.H;
    alpha = params.aalpha;

    %% 1. Enumerate all help vectors h ∈ {0,1}^N
    % Hbin(k,ℓ) = h^ℓ for the k-th help vector (binary order)
    Hbin = dec2bin(0:H-1, N) - '0';    % [H × N]

    %% 2. Build destination-specific help mask
    % Reshape to 1 × N × H so the mask applies to destination ℓ and help index h
    Hmask = permute(reshape(Hbin.', [1, N, H]), [1, 2, 3]);

    %% 3. Compute cost scale factor
    % If h^ℓ = 0 → scale = 1, if h^ℓ = 1 → scale = α
    scale = (1 - Hmask) + alpha * Hmask;   % [1 × N × H]

    %% 4. Apply scaling to base migration costs
    % Broadcast across origins i and help indices h
    tau_eff = params.ttau .* scale;        % [N × N × H]

end
