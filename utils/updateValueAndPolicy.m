function [vf, pol] = updateValueAndPolicy(val, dims, params, grids, indexes, matrices, G, A_current)
% UPDATEVALUEANDPOLICY  Bellman update and optimal policies (assets & migration).
%
%   [VF, POL] = UPDATEVALUEANDPOLICY(VAL, DIMS, PARAMS, GRIDS, INDEXES, MATRICES, G, A_CURRENT)
%   updates value functions for agents without network access (n=0) and with
%   network access (n=1), given current guesses VAL.V and VAL.Vn. It:
%     1) computes continuation values when STAYING via the (z,ψ) transition,
%     2) computes continuation values when MIGRATING via interpolation over a',
%     3) forms migration choice probabilities (softmax/“logit”),
%     4) integrates over destinations and (for n=1) help vectors,
%     5) maximizes over savings on the fine grid to update V and Vn,
%     6) returns migration policy tensors.
%
%   INPUTS
%   ------
%   val      : struct with current value functions
%       .V   [S × K × Na × N]  value without network
%       .Vn  [S × K × Na × N]  value with network
%
%   dims     : struct with model dimensions (S, K, Na, N, H, …)
%   params   : struct with model parameters (bbeta, cchi, CONS, nnu, …)
%   grids    : struct with grids (agrid, ahgrid, …)
%   indexes  : struct with index arrays (II diagonal indices, etc.)
%   matrices : struct with precomputed objects
%       .Ue        [S × K × 1 × N × na] per-period utility on fine grid
%       .a_prime   [S × K × Na × N × N × H] post-migration assets
%       .P         [S × K × K × N] stay operator over (z,ψ)
%   G        : [H × 1] (or H×T) help-PMF; here treated as an exogenous weight
%   A_current: [N × 1] productivity vector for the current period. If omitted
%              the function falls back to MATRICES.Ue (stationary case).
%
%   OUTPUTS
%   -------
%   vf  : struct
%       .R   [S × K × Na × N] expected continuation (n=0)
%       .Rn  [S × K × Na × N] expected continuation (n=1)
%       .V   [S × K × Na × N] updated value (n=0)
%       .Vn  [S × K × Na × N] updated value (n=1)
%
%   pol : struct
%       .a    [S × K × Na × N]  argmax index on fine grid (n=0)
%       .an   [S × K × Na × N]  argmax index on fine grid (n=1)
%       .mu   [S × K × Na × N × N]        migration probs (n=0, single h)
%       .mun  [S × K × Na × N × N × H]    migration probs (n=1, by h)
%
%   NOTES
%   -----
%   • Migration resets to K=1 (U, ψ₁). We therefore interpolate V at K=1
%     for movers, but STAY values retain dependence on current K via P.
%   • “Staying” values are written onto the diagonal (i→i) of the migration
%     value tensor for every help vector h.
%   • Fix for “Attempt to grow array along ambiguous dimension”:
%     assign using linear indices with a column-vector RHS (see below).
%
%   AUTHOR: Agustín Deambrosi
%   LAST REVISED: September 2025
% ======================================================================

    %% 1) Continuation values — if agent stays in place
    % Vmix for networked agents: probability of losing network outside origin.
    % Allow cchi to be scalar, 1×N, or S×N (broadcast to S×1×1×N).
    chi = params.cchi;
    if ~isscalar(chi)
        if isequal(size(chi), [1, dims.N])
            chi = repmat(chi, [dims.S, 1]);
        end
        chi = reshape(chi, [dims.S, 1, 1, dims.N]);  % S×1×1×N
    else
        chi = reshape(chi, [1, 1, 1, 1]);            % scalar broadcast
    end
    Vmix = (1 - chi) .* val.Vn + chi .* val.V;       % S×K×Na×N

    % Apply (z,ψ) transition P(s,:,:,i) to each (s,i) slice
    cont_no_mig     = Pdot(matrices.P, val.V);   % S×K×Na×N
    cont_no_mig_net = Pdot(matrices.P, Vmix);    % S×K×Na×N

    %% 2) Continuation values — if agent migrates (interpolation at a' = a − τ)
    % Base interpolation returns S×Na×N×N×H with K reset to 1 for movers.
    cont_mig_base     = interp_migration(grids.agrid, matrices.a_prime, dims, val.V);
    cont_mig_net_base = interp_migration(grids.agrid, matrices.a_prime, dims, Vmix);

    % Expand movers' values across current K (since migration resets K next period)
    % Target shape for both: S×K×Na×N×N×H
    cont_mig     = repmat(reshape(cont_mig_base,     [dims.S, 1, dims.Na, dims.N, dims.N, dims.H]), [1, dims.K, 1, 1, 1, 1]);
    cont_mig_net = repmat(reshape(cont_mig_net_base, [dims.S, 1, dims.Na, dims.N, dims.N, dims.H]), [1, dims.K, 1, 1, 1, 1]);

    % Overwrite diagonal (i→i) with STAY values for every h (write via linear indexing)
    rhs = repmat(cont_no_mig,     [1, 1, 1, 1, dims.H]);  % S×K×Na×N×H
    cont_mig(indexes.II) = rhs(:);                        % vector assignment

    rhsn = repmat(cont_no_mig_net, [1, 1, 1, 1, dims.H]); % S×K×Na×N×H
    cont_mig_net(indexes.II) = rhsn(:);

    % Infeasible moves (negative a') are disallowed
    cont_mig(matrices.a_prime     < 0) = -Inf;
    cont_mig_net(matrices.a_prime < 0) = -Inf;

    %% 3) Migration probabilities (softmax over destinations ℓ)
    % Shapes: S×K×Na×N×N×H. Normalize along the destination dimension (5th).
    exp_vals     = (exp(cont_mig     / params.CONS)) .^ (1 / params.nnu);
    exp_vals_net = (exp(cont_mig_net / params.CONS)) .^ (1 / params.nnu);

    denom     = sum(exp_vals,     5);
    denom_net = sum(exp_vals_net, 5);

    mmuu     = exp_vals     ./ denom;
    mmuu_net = exp_vals_net ./ denom_net;

    %% 4) Expected continuation value over destinations and help vectors
    % (n = 0) No network: one help vector (take h=1 slice)
    exp_value = mmuu(:,:,:,:,:,1) .* cont_mig(:,:,:,:,:,1);      % S×K×Na×N×N
    exp_value(isnan(exp_value)) = 0;                             % guard numerical artifacts

    % (n = 1) Networked: weight across help vectors with G
    exp_value_net = mmuu_net .* cont_mig_net;                    % S×K×Na×N×N×H
    exp_value_net(isnan(exp_value_net)) = 0;

    % Help weights: broadcast G over S,K,Na,N,N
    Gw = reshape(G, [1, 1, 1, 1, 1, dims.H]);                    % 1×1×1×1×1×H
    Eh = sum(exp_value_net .* Gw, 6);                            % sum over help → S×K×Na×N×N

    %% 5) Compute expected value: R = β · E[V′]
    vf.R  = params.bbeta * sum(exp_value,  5);   % S×K×Na×N
    vf.Rn = params.bbeta * sum(Eh,         5);   % S×K×Na×N

    %% 6) Asset choice: maximize U(c) + R on the fine grid
    if nargin < 8 || isempty(A_current)
        Ue_current = matrices.Ue;
    else
        Ue_current = computeUtilityGivenProductivity(A_current, matrices, indexes.I_Np);
    end

    % (n = 0)
    interp_R          = interpolateToFinerGrid(grids.agrid, grids.ahgrid, vf.R);
    total_val         = Ue_current + interp_R;                 % shapes align: [S×K×1×N×na]
    [vf.V, pol.a]     = max(total_val, [], 5);                  % take argmax over fine asset dim

    % (n = 1)
    interp_Rn         = interpolateToFinerGrid(grids.agrid, grids.ahgrid, vf.Rn);
    total_valn        = Ue_current + interp_Rn;
    [vf.Vn, pol.an]   = max(total_valn, [], 5);

    %% 7) Migration policies
    % For non-network agents, report the h=1 slice (all-zero help vector)
    pol.mu  = mmuu(:,:,:,:,:,1);    % S×K×Na×N×N
    pol.mun = mmuu_net;             % S×K×Na×N×N×H

end
