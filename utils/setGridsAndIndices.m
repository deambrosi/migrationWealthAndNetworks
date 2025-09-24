function [grids, indexes] = setGridsAndIndices(dims)
% SETGRIDSANDINDICES  Construct state grids and index matrices for model solution.
%
%   [GRIDS, INDEXES] = SETGRIDSANDINDICES(DIMS) creates grids for assets
%   and integration, and builds multi-dimensional index arrays used to
%   vectorize value-function iteration, policy evaluation, and simulation.
%
%   INPUT
%   -----
%   dims : struct
%       Model dimensions, must contain:
%         .S   Number of skill types
%         .N   Number of locations
%         .k   Number of integration (ψ) rungs
%         .K   Number of (employment × ψ) states
%         .Na  Number of coarse asset grid points
%         .na  Number of fine asset grid points
%         .H   Number of possible help vectors (2^N)
%
%   OUTPUT
%   ------
%   grids : struct
%       .agrid   Coarse asset grid (non-uniform, Na points)
%       .ahgrid  Fine asset grid   (uniform,   na points)
%       .psi     Integration ladder ψ ∈ [0,1] (length k)
%
%   indexes : struct
%       Index matrices for vectorized computation
%       ---------------------------------------------------------------
%       For value function V (S × K × Na × N):
%         .I_s, I_k, I_a, I_N     Indices for skill, state, assets, location
%
%       For post-saving arrays R (S × K × Na × N × na):
%         .I_sp, I_kp, I_ap, I_Np, I_app
%                                  Same as above, with fine-asset dim
%         .I_ep                    Employment status (1=U, 2=E) from K-index
%         .I_psip                  ψ-index from K-index
%
%       For migration arrays (S × K × Na × N × N × H):
%         .I_sm, I_km, I_am, I_Nm, I_jm, I_hm
%                                  Skill, state, assets, origin, destination, help
%         .II                      Linear indices for “stayers” (origin=destination)
%
%       Sizes:
%         .sz                      Size of main V grid
%         .szp                     Size of post-saving R grid
%         .szm                     Size of migration grid
%
%   DEPENDENCY
%   ----------
%   Requires external function NODEUNIF for grid construction.
%
%   AUTHOR: Agustín Deambrosi
%   LAST REVISED: September 2025
% ======================================================================

    %% 1. Asset grids
    lb.a        = 0;     % Lower bound for assets
    ub.a        = 20;    % Upper bound for assets
    ca          = 3;     % Curvature exponent for coarse grid

    % Coarse grid (agrid): nonlinear spacing for accuracy near zero
    grids.agrid  = nodeunif(dims.Na, 0, (ub.a - lb.a)^(1/ca)).^ca + lb.a;

    % Fine grid (ahgrid): uniform spacing, used for continuous saving policy
    grids.ahgrid = nodeunif(dims.na, lb.a, ub.a);

    %% 2. Integration (ψ) grid
    % Normalized between 0 and 1, with k rungs
    grids.psi    = linspace(0, 1, dims.k);

    %% 3. Indexing for value functions V (S × K × Na × N)
    [I_s, I_k, I_a, I_N] = ndgrid(1:dims.S, 1:dims.K, 1:dims.Na, 1:dims.N);

    %% 4. Indexing for post-saving arrays R (S × K × Na × N × na)
    [I_sp, I_kp, I_ap, I_Np, I_app] = ndgrid( ...
        1:dims.S, 1:dims.K, 1:dims.Na, 1:dims.N, 1:dims.na);

    % Recover employment and ψ from packed K-index
    I_ep   = floor((I_kp - 1) / dims.k) + 1;   % Employment (1=U, 2=E)
    I_psip = mod(I_kp - 1, dims.k) + 1;        % ψ-rung (1..k)

    %% 5. Migration-related indexing (S × K × Na × N × H)
    [I_sr, I_kr, I_ar, I_ir, I_hr] = ndgrid( ...
        1:dims.S, 1:dims.K, 1:dims.Na, 1:dims.N, 1:dims.H);

    % Linear indices for “stayers” (origin=destination, diagonal in N)
    II = sub2ind([dims.S, dims.K, dims.Na, dims.N, dims.N, dims.H], ...
                 I_sr, I_kr, I_ar, I_ir, I_ir, I_hr);

    %% 6. Full migration indexing (S × K × Na × N × N × H)
    [I_sm, I_km, I_am, I_Nm, I_jm, I_hm] = ndgrid( ...
        1:dims.S, 1:dims.K, 1:dims.Na, 1:dims.N, 1:dims.N, 1:dims.H);

    %% 7. Sizes of main index structures
    sz  = size(I_s);     % For V
    szp = size(I_sp);    % For R
    szm = size(I_sm);    % For migration

    %% 8. Pack outputs
    indexes.I_s    = I_s;
    indexes.I_k    = I_k;
    indexes.I_a    = I_a;
    indexes.I_N    = I_N;

    indexes.I_sp   = I_sp;
    indexes.I_kp   = I_kp;
    indexes.I_ap   = I_ap;
    indexes.I_Np   = I_Np;
    indexes.I_app  = I_app;
    indexes.I_ep   = I_ep;
    indexes.I_psip = I_psip;

    indexes.I_sm   = I_sm;
    indexes.I_km   = I_km;
    indexes.I_am   = I_am;
    indexes.I_Nm   = I_Nm;
    indexes.I_jm   = I_jm;
    indexes.I_hm   = I_hm;

    indexes.II     = II;

    indexes.sz     = sz;
    indexes.szp    = szp;
    indexes.szm    = szm;

end
