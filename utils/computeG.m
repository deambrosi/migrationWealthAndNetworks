function G = computeG(M, ggamma)
% COMPUTEG  Compute PMF over help-offer vectors h ∈ {0,1}^N.
%
%   G = COMPUTEG(M, γ) evaluates the probability mass function of the help
%   vector h = (h¹,…,hᴺ), where each component is an independent Bernoulli
%   random variable with success probability πʲ(Mʲ_t) = (Mʲ_t)^γ.
%
%   INPUTS
%   ------
%   M       : [N × T] matrix
%             Migrant network masses per location j=1..N and time t=1..T.
%   ggamma  : scalar
%             Elasticity parameter γ > 0 governing how help probability
%             scales with migrant network mass.
%
%   OUTPUT
%   -------
%   G       : [2^N × T] matrix
%             Each row corresponds to one help vector h (in binary order).
%             Column t is the probability distribution g(h | M_t).
%
%   NOTES
%   -----
%   • Help vectors are enumerated in binary order by rows of Hmat.
%   • Independence assumption: arrivals of help to each destination j are
%     independent conditional on M_t.
%   • Rows of G(:,t) sum to one for each t (within numerical tolerance).
%
%   AUTHOR: Agustín Deambrosi
%   LAST REVISED: September 2025
% ======================================================================

    %% 1. Dimensions and probabilities -----------------------------------
    [N, T] = size(M);

    % Success probabilities for each location j at each time t
    % π^j(M^j_t) = (M^j_t)^γ
    P = M .^ ggamma;   % [N × T]

    % Enumerate all possible help vectors h ∈ {0,1}^N
    Hmat = dec2bin(0:(2^N - 1)) - '0';   % [2^N × N]

    % Preallocate output PMF
    G = zeros(2^N, T);

    %% 2. Loop over time periods -----------------------------------------
    for t = 1:T
        pi_t      = P(:, t)';           % [1 × N] probabilities at time t
        one_minus = 1 - pi_t;           % Complement probabilities

        % Loop over help-vector indices
        for h_idx = 1:2^N
            h = Hmat(h_idx, :);         % 1 × N binary help vector

            % Probability mass: ∏_j [π_j^h_j * (1-π_j)^(1-h_j)]
            G(h_idx, t) = prod(pi_t.^h .* one_minus.^(1 - h));
        end
    end

end
