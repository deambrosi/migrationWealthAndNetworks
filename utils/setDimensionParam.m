function dims = setDimensionParam()
% SETDIMENSIONPARAM  Initialize and return model dimension parameters.
%
%   DIMS = SETDIMENSIONPARAM() returns a struct with all fixed dimension sizes
%   used throughout the model (state space sizes, grids, etc.). These values
%   define array shapes for value/policy functions and transition objects.
%
%   OUTPUT
%   ------
%   dims : struct with fields
%       .S   Number of skill types.
%       .N   Number of locations (destinations), with 1 reserved for origin.
%       .k   Number of country-specific human capital (integration) rungs.
%       .K   Total (employment × integration) states per location:
%            K = 2 * k, stacking first Unemployed(ψ=1..k) then Employed(ψ=1..k).
%       .H   Number of possible help-offer vectors: H = 2^N.
%       .Na  Number of coarse asset grid points (on agrid).
%       .na  Number of fine asset grid points (on ahgrid).
%
%   CONVENTIONS
%   -----------
%   • Employment × Integration packing:
%       - The second state dimension uses K = 2*k rows.
%       - Rows 1..k   correspond to Unemployed with ψ = 1..k.
%       - Rows k+1..2k correspond to Employed   with ψ = 1..k.
%   • Help vectors:
%       - There are H = 2^N binary help vectors, one component per destination.
%       - Indexing of the help dimension (when present) is 1..H.
%
%   AUTHOR
%   ------
%   Agustín Deambrosi
%
%   LAST REVISED
%   ------------
%   September 2025

    % -----------------------------
    % Core model dimension settings
    % -----------------------------
    dims.S  = 1;                % Number of skill types
    dims.N  = 6;                % Number of locations
    dims.k  = 5;                % # of ψ (integration) rungs per employment state

    % Derived counts
    dims.K  = dims.k * 2;       % Total (employment × ψ) states per location
    dims.H  = 2 ^ dims.N;       % # of possible help-offer combinations

    % Asset grids
    dims.Na = 50;               % Coarse asset grid size (on agrid)
    dims.na = 5000;             % Fine  asset grid size (on ahgrid)

end

