function tau_h = getMigrationCostsWithHelp(tau, alpha)
% GETMIGRATIONCOSTSWITHHELP Computes adjusted migration costs with help.
%
%   Given the baseline migration cost matrix tau(i,j) and a cost-reduction
%   factor alpha, this function returns a 3D tensor tau_h(i,j,h), where 
%   h indexes over the 2^N possible binary help vectors. If help is offered 
%   from location j (i.e., h(j) = 1), the cost of migrating to j is scaled 
%   by alpha. Otherwise, it remains as tau(i,j).
%
%   INPUTS:
%       tau     - [N x N] matrix of baseline migration costs
%       alpha   - Cost-reduction factor for helped moves (0 < alpha < 1)
%
%   OUTPUT:
%       tau_h   - [N x N x 2^N] tensor of migration costs indexed by help vector
%
%   AUTHOR: Agustin Deambrosi
%   LAST REVISED: April 2025
% =========================================================================

    N			= size(tau, 1);              % Number of locations
    H			= 2^N;                       % Number of possible help vectors
    hmat		= dec2bin(0:H-1) - '0';      % [H x N] matrix of all binary help vectors

    tau_h		= repmat(tau, 1, 1, H);      % Initialize tensor with baseline costs

    for h_idx = 1:H
        h_vec = hmat(h_idx, :);             % Help vector for this configuration
        for j = 1:N
            if h_vec(j)
                tau_h(:, j, h_idx) = alpha * tau(:, j);  % Apply discount to destination j
            end
        end
    end

end
