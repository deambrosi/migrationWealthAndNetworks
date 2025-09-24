function Y = Pdot(P, X)
% PDOT  Apply K×K transition blocks to K×Na value slices without loops.
%
%   Y = PDOT(P, X) computes, for every (skill s, location i), the product
%   P(s,:,:,i) * X(s,:,:,i), where the packed K-dimension indexes the
%   joint (employment × ψ) state. The location index i is the LAST
%   dimension in all inputs/outputs.
%
%   INPUTS
%   ------
%   P : S × K × K × N
%       Transition operator for the packed state when STAYING in location i.
%       For each (s,i), P(s,:,:,i) must be K×K and row-stochastic.
%
%   X : S × K × Na × N
%       Array to which the transition is applied along K (e.g., a value
%       function or continuation term). For each (s,i), X(s,:,:,i) is K×Na.
%
%   OUTPUT
%   ------
%   Y : S × K × Na × N
%       Result of blockwise multiplication with i as LAST dim:
%           Y(s,:,:,i) = P(s,:,:,i) * X(s,:,:,i).
%
%   NOTES
%   -----
%   • No explicit loops: we reshape/permute to use a single pagemtimes call.
%   • Location i remains the last dimension before and after multiplication.
%   • Requirements: MATLAB R2020b+ for pagemtimes.
%
%   EXAMPLE
%   -------
%       % P: S×K×K×N,  V: S×K×Na×N
%       cont = Pdot(P, V);
%
%   AUTHOR: Agustín Deambrosi
%   LAST REVISED: September 2025
% ======================================================================

    % ----- dimension checks
    [S,  K1, K2, N]  = size(P);
    [Sx, Kx, Na, Nx] = size(X);

    assert(S == Sx, 'Pdot: skill dimension mismatch (P:%d vs X:%d).', S, Sx);
    assert(N == Nx, 'Pdot: location dimension mismatch (P:%d vs X:%d).', N, Nx);
    assert(K1 == K2, 'Pdot: P must be square in K (got %d×%d).', K1, K2);
    assert(K1 == Kx, 'Pdot: K mismatch (P uses %d, X uses %d).', K1, Kx);

    K = K1;

    % ----- bring K to front, batch = (s,i) with i as LAST on reshape back
    % P_f : K × K  × (S*N)   (pages: s fastest, i slowest)
    % X_f : K × Na × (S*N)
    P_f = reshape(permute(P, [2 3 1 4]), [K, K,  S*N]);
    X_f = reshape(permute(X, [2 3 1 4]), [K, Na, S*N]);

    % ----- pagewise multiply: for each (s,i) page, do (K×K) * (K×Na)
    Y_f = pagemtimes(P_f, X_f);   % K × Na × (S*N)

    % ----- restore to S × K × Na × N, preserving i as LAST dim
    Y = permute(reshape(Y_f, [K, Na, S, N]), [3 1 2 4]);
end
