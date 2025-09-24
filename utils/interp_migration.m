function f = interp_migration(agrid, a_prime, dims, V)
% INTERP_MIGRATION  Interpolate continuation values at post-migration assets.
%
%   F = INTERP_MIGRATION(AGRID, A_PRIME, DIMS, V) evaluates the value function
%   over asset levels *after* paying migration costs, for every origin–destination
%   pair and help vector. This provides the continuation value term used in the
%   migration choice stage.
%
%   INPUTS
%   ------
%   agrid   : [Na × 1]
%       Coarse asset grid (monotone; interpolation domain).
%
%   a_prime : [S × K × Na × N × N × H]
%       Post-migration asset levels a' = a − τ^{iℓ}(h) laid out across
%       (skill, packed state K, asset grid, origin i, destination ℓ, help h).
%
%   dims    : struct
%       Dimension parameters (Na, N, H, S, K, …).
%
%   V       : [S × K × Na × N]
%       Value function evaluated *before* migration, indexed by
%       (skill, packed state K = employment×ψ, assets, location).
%
%   OUTPUT
%   -------
%   f       : [S × Na × N × N × H]
%       Interpolated continuation values at post-migration assets a' for
%       each (skill, asset grid point, origin i, destination ℓ, help h).
%
%   NOTES
%   -----
%   • The interpolation is performed holding the *packed* state fixed at K=1
%     (conventionally (U, ψ₁)), consistent with the model’s reset upon migration.
%   • Interpolation is linear in assets and uses 'extrap' to guard boundary cases.
%   • The function is vectorized by flattening batches and using PARFOR over
%     destination/help configurations.
%
%   AUTHOR: Agustín Deambrosi
%   LAST REVISED: September 2025
% ======================================================================

    %% 1) Reorder and compress V for interpolation
    % Bring assets to the leading dimension: V_reordered : [Na × S × K × N]
    V_reordered = permute(V, [3, 1, 2, 4]);

    % Remove the K-dimension by selecting the reset state after migration:
    % K=1 corresponds to (U, ψ₁) under the packed ordering.
    % Result: V_base : [Na × S × N]
    V_base = squeeze(V_reordered(:, :, 1, :));

    % Expand V over (destination, help) to match the evaluation grid later:
    % V_expanded : [Na × S × N(origin) × N(dest) × H]
    V_expanded = repmat(V_base, 1, 1, 1, dims.N, dims.H);

    % Flatten batches to columns so we can interpolate all at once:
    % V_flat : [Na × (S·N·N·H)]
    V_flat = reshape(V_expanded, dims.Na, []);

    %% 2) Align a' (post-migration assets) with the same flattened batch
    % Bring assets to the leading dimension:
    % a_prime : [Na × S × K × N × N × H]
    a_prime = permute(a_prime, [3, 1, 2, 4, 5, 6]);

    % Remove the packed-state dimension (use K=1 consistent with reset):
    % a_prime : [Na × S × N × N × H]
    a_prime = squeeze(a_prime(:, :, 1, :, :, :));

    % Flatten to align with columns of V_flat:
    % a_flat : [Na × (S·N·N·H)]
    a_flat = reshape(a_prime, dims.Na, []);

    %% 3) Interpolate in parallel across all (s,i,ℓ,h) configurations
    f_interp = zeros(size(V_flat));  % preallocate [Na × batches]
    parfor j = 1:size(V_flat, 2)
        % Linear interpolation (with extrapolation safeguard at boundaries)
        f_interp(:, j) = interp1(agrid, V_flat(:, j), a_flat(:, j), 'linear', 'extrap');
    end

    %% 4) Reshape back to target shape [S × Na × N × N × H]
    % First unflatten to [Na × S × N × N × H]
    f_shaped = reshape(f_interp, dims.Na, dims.S, dims.N, dims.N, dims.H);

    % Then permute to [S × Na × N × N × H]
    f = permute(f_shaped, [2, 1, 3, 4, 5]);

end
