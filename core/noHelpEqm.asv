function [vf, pol] = noHelpEqm(dims, params, grids, indexes, matrices, settings)
% NOHELPEQM Solves the value function with no migration help (G = G0).
%
%   Solves for the value functions V (with network) and Vn (without network)
%   assuming a stationary environment with no help offers from the migrant network.
%   It performs value function iteration until convergence.
%
%   INPUTS:
%       dims     - Model dimensions (struct)
%       params   - Model parameters (struct)
%       grids    - Grid structures (struct)
%       indexes  - Precomputed index matrices (struct)
%       matrices - Precomputed utility and wealth matrices (struct)
%       settings - Iteration settings and tolerances (struct)
%
%   OUTPUTS:
%       vf   - Struct containing converged value functions (.V, .Vn)
%       pol  - Struct containing optimal policies
%
%   AUTHOR: Agustin Deambrosi
%   LAST REVISED: April 2025
% =========================================================================

    %% Initialization
    diffV		= 1;
    itV			= 0;

    % Initialize value functions uniformly
    val.V		= ones(indexes.sz);     % With network
    val.Vn		= val.V;                % Without network

    %% Value Function Iteration (no help offers)
    while (diffV > settings.tolV) && (itV < settings.MaxItJ)
        itV		= itV + 1;

        [vf, pol] = updateValueAndPolicy(val, dims, params, grids, indexes, matrices, params.G0);

        % Update convergence criterion
        diffV	= sum(abs(vf.Vn - val.Vn) ./ (1 + abs(vf.Vn)), 'all');
        val		= vf;
    end

end
