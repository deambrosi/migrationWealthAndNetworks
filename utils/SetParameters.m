function params = SetParameters(dims)
% SETPARAMETERS  Initialize structural parameters used in the migration model.
%
%   PARAMS = SETPARAMETERS(DIMS) creates a struct with all parameter values
%   that govern preferences, location-specific features, employment dynamics,
%   migration frictions, and network effects.
%
%   INPUT
%   -----
%   dims : struct
%       Contains dimension settings (fields: S, N, k, K, H, Na, na).
%
%   OUTPUT
%   ------
%   params : struct with fields
%
%   Preferences
%     .bbeta     Discount factor (quarterly).
%     .xi        Weight on host-country human capital in utility.
%     .phiH      Elasticity of utility with respect to human capital.
%
%   Location-Specific Features
%     .A         Productivity levels by location (vector length N).
%     .B         Amenity scales by location.
%     .theta_s   Skill premia by skill × location (S×N matrix).
%     .theta_k   Elasticity of human capital in earnings.
%     .bbi       Unemployment income by location (N×1 vector).
%     .f         Endpoints of job-finding probabilities
%                (S×N×2 array: [ψ=0, ψ=1]).
%     .g         Endpoints of job-separation probabilities
%                (S×N×2 array: [ψ=0, ψ=1]).
%     .up_psi    Probability that integration ψ increases by one rung
%                each period (vector length N).
%
%   Migration & Choice Frictions
%     .ttau      Base migration cost matrix (N×N, asymmetric).
%     .nnu       Scale of i.i.d. location taste shocks (logit scale).
%
%   Network Help Mechanics
%     .aalpha    Fractional cost when help is received (0<alpha<1).
%     .ggamma    Elasticity of help probability w.r.t. network mass.
%     .cchi      Probability that network affiliation decays each period
%                outside Venezuela.
%     .G0        Help-offer distribution when M=0 (all-zero vector).
%
%   Calibration
%     .CONS      Scaling constant for calibration.
%
%   AUTHOR: Agustín Deambrosi
%   LAST REVISED: September 2025
% ======================================================================

    %% Preferences
    params.bbeta   = 0.996315;   % Discount factor (quarterly)
    params.xi      = 1.5;        % Utility weight on human capital ψ
    params.phiH    = 0.3;        % Elasticity of utility w.r.t. ψ

    %% Location-specific features
    params.A       = (1:dims.N)';         % Productivity by location (wage shifters)
    params.B       = ones(dims.N, 1);     % Amenities by location

    % Skill premia: expand across S×N
    params.theta_s = repmat(1:dims.N, dims.S, 1);  
    params.theta_k = 2;                   % Returns to human capital in wages

    % Income flows
    params.bbi     = 0.2 * ones(dims.N, 1);  % Unemployment income (N×1)

    % Job-finding probabilities (endpoints at ψ=0 and ψ=1)
    f_base         = [0.2*ones(dims.N,1), 0.9*ones(dims.N,1)];   % N×2
    f_base         = permute(f_base, [3,1,2]);                   % 1×N×2
    params.f       = repmat(f_base, [dims.S, 1, 1]);             % S×N×2

    % Job-separation probabilities (endpoints at ψ=0 and ψ=1)
    g_base         = [0.5*ones(dims.N,1), 0.02*ones(dims.N,1)];  % N×2
    g_base         = permute(g_base, [3,1,2]);                   % 1×N×2
    params.g       = repmat(g_base, [dims.S, 1, 1]);             % S×N×2

    % Integration progression (Markov upward transition probability per period)
    params.up_psi  = 0.15 * ones(dims.N, 1);  

    %% Migration & choice frictions
    % Base migration cost matrix (encourages stepping-stone migration).
    params.ttau = [ 0   3   8  12  15;   % From 1 -> {2,3,4,5}
                    3   0   3   8  12;   % From 2 -> {1,3,4,5}
                    8   3   0   3   8;   % From 3 -> {1,2,4,5}
                   12   8   3   0   3;   % From 4 -> {1,2,3,5}
                   15  12   8   3   0];  % From 5 -> {1,2,3,4}

    params.nnu    = 0.1;   % Scale of i.i.d. taste shocks (logit)

    %% Help mechanics (network effects)
    params.aalpha = 0.2;   % Fractional migration cost when helped
    params.ggamma = 2;     % Elasticity of help probability
    params.cchi   = 0.15;  % Network erosion probability outside Venezuela

    % Help distribution at zero migrant masses (all-zero help vector)
    params.G0     = computeG(zeros(dims.N,1), params.ggamma);  

    %% Calibration constant
    params.CONS   = 1e2;   % Scaling constant for calibration

end
