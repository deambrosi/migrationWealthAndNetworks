function settings = IterationSettings()
% ITERATIONSETTINGS  Initialize iteration and simulation parameters.
%
%   SETTINGS = ITERATIONSETTINGS() creates a struct with default values
%   for iteration counters, convergence tolerances, loop limits, and 
%   simulation configuration used in solving and simulating the model.
%
%   OUTPUT
%   ------
%   settings : struct with fields
%
%   Iteration controls
%     .it        Iteration counter (initialized at 0).
%     .diffV     Initial maximum difference in the value function (set to 1).
%
%   Convergence tolerances
%     .tolV      Tolerance threshold for value function convergence.
%     .tolM      Tolerance threshold for migration convergence.
%
%   Iteration limits
%     .MaxItV    Maximum number of iterations for value function iteration.
%     .MaxItJ    Maximum number of iterations for policy improvement / inner loops.
%     .MaxIter   Maximum number of iterations for the outer equilibrium algorithm.
%
%   Simulation settings
%     .Nagents   Number of simulated agents.
%     .T         Number of simulated time periods.
%     .burn      Number of initial burn-in periods dropped from analysis.
%
%   AUTHOR
%   ------
%   Agustín Deambrosi
%
%   LAST REVISED
%   ------------
%   September 2025
% ======================================================================

    %% Iteration counters and initialization
    settings.it      = 0;   % Iteration counter
    settings.diffV   = 1;   % Initial V-function difference

    %% Convergence tolerances
    settings.tolV    = 0.5;     % Value function convergence threshold
    settings.tolM    = 1e-2;    % Migration convergence threshold

    %% Iteration limits
    settings.MaxItV  = 40;      % Maximum iterations for value function iteration
    settings.MaxItJ  = 10;      % Maximum iterations for policy update (inner loop)
    settings.MaxIter = 100;     % Maximum iterations for outer algorithm

    %% Simulation configuration
    settings.Nagents = 5000;    % Number of simulated agents
    settings.T       = 100;     % Total simulated time periods
    settings.burn    = 50;      % Burn-in periods removed from statistics

end
