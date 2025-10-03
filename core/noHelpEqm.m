function [vf, pol] = noHelpEqm(dims, params, grids, indexes, matrices, settings)
% NOHELPEQM  Solve value functions under no-help environment (G = G0).
%
%   [VF, POL] = NOHELPEQM(DIMS, PARAMS, GRIDS, INDEXES, MATRICES, SETTINGS)
%   performs value-function iteration for the stationary case with *no*
%   migration help offers from the network (i.e., help PMF fixed at G0).
%   It returns the converged value functions—with network (V) and without
%   network (Vn)—and the associated optimal policies.
%
%   INPUTS
%   ------
%   dims     : struct
%       Model dimensions (S, K, Na, N, H, …).
%
%   params   : struct
%       Model parameters. Must contain:
%         .G0   H×1 or H×T help PMF evaluated at M=0 (used here as fixed).
%
%   grids    : struct
%       Grids for assets and other state components (e.g., agrid, ahgrid).
%
%   indexes  : struct
%       Precomputed index arrays and size references (see setGridsAndIndices).
%
%   matrices : struct
%       Precomputed static matrices (e.g., Ue, a_prime, P, mig_costs).
%
%   settings : struct
%       Iteration and simulation controls. Must contain:
%         .tolV    Convergence tolerance for V iteration
%         .MaxItJ  Maximum iterations for this inner loop
%
%   OUTPUTS
%   -------
%   vf  : struct
%       .V   Value function with network access     [S × K × Na × N]
%       .Vn  Value function without network access  [S × K × Na × N]
%
%   pol : struct
%       Optimal policies returned by UPDATEVALUEANDPOLICY (e.g., savings,
%       migration, and related objects as defined there).
%
%   NOTES
%   -----
%   • This routine fixes the help distribution at G0 and iterates on the
%     value functions only. Network masses and equilibrium consistency are
%     handled elsewhere in the full solution pipeline.
%   • Convergence metric is a scale-free L1 change:
%       sum( |Vn_new - Vn_old| ./ (1 + |Vn_new|) ).
%
%   AUTHOR: Agustín Deambrosi
%   LAST REVISED: September 2025
% ======================================================================

    %% 1) Initialization
    diffV = 1;
    itV   = 0;

    % Start from a simple uniform guess (same shape as V/Vn)
    val.V  = ones(indexes.sz, 'like', matrices.Ue);  % With network
    val.Vn = val.V;                                  % Without network

    %% 2) Value Function Iteration (no help offers: G = G0)
    while (diffV > settings.tolV) && (itV < settings.MaxItJ)
        itV = itV + 1;

        % Bellman update and policy extraction given fixed help PMF G0
        [vf, pol] = updateValueAndPolicy(val, dims, params, grids, indexes, matrices, params.G0, params.A);

        % Scale-free convergence check on Vn (robust when levels change)
        diffV = sum(abs(vf.Vn - val.Vn) ./ (1 + abs(vf.Vn)), 'all');

        % Update iterate
        val = vf;
    end

end
