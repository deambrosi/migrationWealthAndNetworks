function P = build_P(dims, params)
% BUILD_P  Joint (z, ψ) transition when STAYING in location i.
%
%   P = BUILD_P(DIMS, PARAMS) constructs the conditional transition operator
%   for the packed employment×integration state when an agent remains in the
%   same location. The operator is skill- and location-specific and has the
%   shape S × K × K × N. Rows index the CURRENT K-state; columns index the
%   NEXT K-state; thus each S×N slice is row-stochastic over the 3rd dim.
%
%   INPUTS
%   ------
%   dims : struct
%       .S   Number of skill types
%       .k   Number of integration (ψ) rungs (B)
%       .K   Number of packed states K = 2*B (U(1..B), E(1..B))
%       .N   Number of locations
%
%   params : struct
%       .f        S × N × 2 endpoints of job-finding: [ψ=0, ψ=1]
%       .g        S × N × 2 endpoints of separation : [ψ=0, ψ=1]
%       .up_psi   N×1 or S×N probability of ψ increasing by one rung
%
%   OUTPUT
%   ------
%   P : S × K × K × N
%       For each skill s and location i, P(s,:,:,i) is a K×K transition
%       matrix with:
%         • From (U, ψ_b): U→U with prob (1−f_i^s(ψ_b)), U→E with prob f_i^s(ψ_b);
%         • From (E, ψ_b): E→U with prob q_i^s(ψ_b),     E→E with prob (1−q_i^s(ψ_b));
%         • ψ moves up one rung with prob up_psi and otherwise stays; top rung
%           is absorbing for ψ. Rows sum to 1 (numerically enforced).
%
%   CONVENTIONS
%   -----------
%   • Packed K ordering: first B rows/cols are Unemployed (U, ψ=1..B),
%     next B rows/cols are Employed (E, ψ=1..B).
%   • f(ψ) is strictly increasing in ψ; q(ψ) strictly decreasing in ψ.
%     We interpolate between endpoints (ψ=0 and ψ=1) with an anchored-logit
%     schedule over the normalized ψ-grid in [0,1].
%
%   AUTHOR: Agustín Deambrosi
%   LAST REVISED: September 2025
% ======================================================================

    % ---- dimensions
    S = dims.S; 
    B = dims.k; 
    K = dims.K; 
    N = dims.N;

    % ---- allocate output
    P = zeros(S, K, K, N);

    % ---- ψ grid in [0,1] with B points
    psi = reshape(linspace(0, 1, B), [1 1 B]);   % 1×1×B

    % ---- endpoints (S×N)
    fL = squeeze(params.f(:,:,1));   % job-finding at ψ=0
    fH = squeeze(params.f(:,:,2));   % job-finding at ψ=1
    qH = squeeze(params.g(:,:,1));   % separation  at ψ=0 (high)
    qL = squeeze(params.g(:,:,2));   % separation  at ψ=1 (low)

    % ---- numerically safe logit/sigmoid
    epsp  = 1e-10;
    clip  = @(x) min(max(x, epsp), 1 - epsp);
    logit = @(p) log(clip(p) ./ (1 - clip(p)));
    sigm  = @(x) 1 ./ (1 + exp(-x));

    % ---- anchored-logit schedules f_i^s(ψ), q_i^s(ψ) on the B grid
    aF = logit(fL);  bF = logit(fH) - aF;   % S×N
    aQ = logit(qH);  bQ = logit(qL) - aQ;   % S×N (typically negative)

    % broadcast to S×N×B
    aF3 = repmat(aF, [1 1 B]);
    bF3 = repmat(bF, [1 1 B]);
    aQ3 = repmat(aQ, [1 1 B]);
    bQ3 = repmat(bQ, [1 1 B]);
    psi3 = repmat(psi, [S N 1]);

    fpsi = sigm(aF3 + bF3 .* psi3);   % S×N×B (increasing in ψ)
    qpsi = sigm(aQ3 + bQ3 .* psi3);   % S×N×B (decreasing in ψ)

    % ---- ψ transition (up with prob up_psi; top rung absorbing)
    up = params.up_psi;
    if isvector(up) && numel(up) == N
        up = reshape(up, [1 N]);      % 1×N
        up = repmat(up, [S 1]);       % S×N
    else
        assert(all(size(up) == [S N]), 'up_psi must be N×1 or S×N');
    end

    % helper for ψ transition row (1×B) from rung b
    function trow = psi_row(ui, b)
        trow = zeros(1, B);
        if b < B
            trow(b)   = 1 - ui;
            trow(b+1) = ui;
        else
            trow(B)   = 1;            % top rung stays
        end
    end

    % packed indices
    Uidx = 1:B;
    Eidx = B + (1:B);

    % ---- fill P(s,:,:,i) for each skill and location
    for s = 1:S
        for i = 1:N
            ui = up(s, i);

            % prebuild ψ-transition rows for (s, i)
            Trows = zeros(B, B);
            for b = 1:B
                Trows(b, :) = psi_row(ui, b);
            end

            % fill rows for each ψ rung b
            for b = 1:B
                fu   = fpsi(s, i, b);   % Pr(find job | U, ψ_b, i, s)
                qu   = qpsi(s, i, b);   % Pr(separate | E, ψ_b, i, s)
                trow = Trows(b, :);     % 1×B over ψ_{t+1}

                % From (U, ψ_b):
                P(s, Uidx(b), Uidx, i) = (1 - fu) * trow;  % stay U, ψ evolves
                P(s, Uidx(b), Eidx, i) = (    fu) * trow;  % find E, ψ evolves

                % From (E, ψ_b):
                P(s, Eidx(b), Uidx, i) = (    qu) * trow;  % separate to U
                P(s, Eidx(b), Eidx, i) = (1 - qu) * trow;  % keep E
            end

            % numerical tidy-up: enforce row-stochasticity
            rowsums = sum(P(s, :, :, i), 3);
            drift   = abs(rowsums - 1) > 1e-12;
            if any(drift)
                P(s, :, :, i) = P(s, :, :, i) ./ rowsums;
            end
        end
    end
end
