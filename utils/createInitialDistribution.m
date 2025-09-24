function m0 = createInitialDistribution(dims, settings)
% CREATEINITIALDISTRIBUTION  Generate initial distribution of agents across states.
%
%   M0 = CREATEINITIALDISTRIBUTION(DIMS, SETTINGS) returns a struct array
%   of length SETTINGS.Nagents with randomized starting positions for each
%   agent on skill, state (employment × ψ), and wealth. All agents begin
%   in Venezuela (location = 1) and are initially network-affiliated.
%
%   INPUTS
%   ------
%   dims : struct
%       Model dimensions, must contain:
%         .S   Number of skill types
%         .K   Number of (employment × ψ) states
%         .Na  Number of coarse asset grid points
%         .N   Number of locations (used for .location initialization)
%
%   settings : struct
%       Simulation settings, must contain:
%         .Nagents  Number of agents to initialize
%
%   OUTPUT
%   ------
%   m0 : [Nagents × 1] struct array with fields
%       .skill     Skill type index (1..S)
%       .state     Packed index of employment × ψ state (1..K)
%       .wealth    Index on coarse asset grid (1..Na)
%       .location  Location index (all start at 1 = Venezuela)
%       .network   Network affiliation (all start as 1 = in network)
%
%   NOTES
%   -----
%   • This routine sets the initial distribution at t=0 for simulations.
%   • Random draws are uniform over the admissible ranges of skill, state,
%     and wealth. You may replace these with empirical distributions later.
%
%   AUTHOR: Agustín Deambrosi
%   LAST REVISED: September 2025
% ======================================================================

    numAgents = settings.Nagents;

    % Preallocate struct array with default values
    m0 = repmat(struct( ...
        'skill',    0, ...
        'state',    0, ...
        'wealth',   0, ...
        'location', 0, ...
        'network',  0), numAgents, 1);

    % Initialize each agent
    for i = 1:numAgents
        m0(i).skill    = randi(dims.S);   % Random skill type
        m0(i).state    = randi(dims.K);   % Random (employment × ψ) state
        m0(i).wealth   = randi(dims.Na);  % Random asset index on coarse grid
        m0(i).location = 1;               % All start in Venezuela (location 1)
        m0(i).network  = 1;               % All start network-affiliated
    end

end
