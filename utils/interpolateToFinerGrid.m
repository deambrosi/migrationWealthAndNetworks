function f = interpolateToFinerGrid(grid, xgrid, Raux)
% INTERPOLATETOFINERGRID  Interpolate value matrices from a coarse to a finer asset grid.
%
%   F = INTERPOLATETOFINERGRID(GRID, XGRID, RAUX) linearly interpolates a value
%   array defined on a coarse asset grid onto a finer asset grid, preserving all
%   non-asset dimensions. This is used before optimizing over savings on the
%   fine grid.
%
%   INPUTS
%   ------
%   grid   : [Na × 1] (or 1×Na)
%       Coarse asset grid (e.g., agrid). Must be monotone increasing.
%
%   xgrid  : [na × 1] (or 1×na)
%       Finer asset grid (e.g., ahgrid). Must lie within or near GRID’s range;
%       points outside are handled via 'extrap'.
%
%   Raux   : value array with asset on its 3rd dimension
%       Typical shape in this codebase: [S × K × Na × N] or [S × K × Na × N × …]
%       (only the 3rd dimension—assets—is interpolated; others are preserved).
%
%   OUTPUT
%   -------
%   f      : [S × K × 1 × N × na]
%       Interpolated values on the fine grid, with a singleton 3rd dim kept
%       for compatibility with downstream maximization code. If RAUX had
%       extra trailing dims, these are preserved in the same relative order.
%
%   NOTES
%   -----
%   • Interpolation method is 'linear' with 'extrap' at the boundaries.
%   • The function flattens all non-asset dimensions, interpolates each
%     column independently, and reshapes back.
%   • No changes to numerical behavior—only reordering for interpolation
%     and restoring the original structure (with a singleton dim for assets).
%
%   AUTHOR: Agustín Deambrosi
%   LAST REVISED: September 2025
% ======================================================================

    %% 1) Bring the asset dimension to the front for interpolation
    % Input RAUX is assumed [S × K × Na × N × …]; we permute to:
    %   Raux_perm : [Na × S × K × N × …]
    Raux = permute(Raux, [3, 1, 2, 4, 5]);

    %% 2) Flatten all non-asset dimensions to columns
    % Save original sizes, then reshape to [Na × batch]
    sz    = size(Raux);                  % sz = [Na, S, K, N, …]
    Rflat = reshape(Raux, sz(1), []);    % Na × (S*K*N*…)

    %% 3) Interpolate on the fine grid
    % Result has size [na × batch]
    Rinterp = interp1(grid, Rflat, xgrid, 'linear', 'extrap');

    %% 4) Reshape back, replacing Na with na
    sz(1)     = length(xgrid);           % swap Na → na
    Rreshaped = reshape(Rinterp, sz);    % [na × S × K × N × …]

    %% 5) Reorder to final layout expected by maximization:
    %   f : [S × K × 1 × N × na]
    f = permute(Rreshaped, [2, 3, 5, 4, 1]);

end
