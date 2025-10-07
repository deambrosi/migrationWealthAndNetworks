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
%   • Skill and wealth draws follow configurable discrete distributions that
%     place 20 percent of agents in the high-skill group by default and bias
%     wealth toward lower asset levels, with higher wealth for skilled agents.
%     Override the defaults via settings.initialDist.
%
%   AUTHOR: Agustín Deambrosi
%   LAST REVISED: September 2025
% ======================================================================

    numAgents = settings.Nagents;

    % ------------------------------------------------------------------
    % Parameterization of the initial distribution
    % ------------------------------------------------------------------
    % Built-in defaults (can be overwritten via settings.initialDist)
    initDefaults = struct( ...
        'shareSkilled',            0.20, ...   % Share of high-skill agents
        'wealthDecayLowSkill',     0.00, ...   % Higher value => more mass at low wealth
        'wealthDecayHighSkill',    0.00);      % Lower value => flatter wealth distribution

    if isfield(settings, 'initialDist') && isstruct(settings.initialDist)
        userParams = settings.initialDist;
        defaultFields = fieldnames(initDefaults);
        for f = 1:numel(defaultFields)
            fieldName = defaultFields{f};
            if isfield(userParams, fieldName)
                initDefaults.(fieldName) = userParams.(fieldName);
            end
        end
    end

    % Clamp shares/parameters to admissible ranges
    shareSkilled = max(0, min(1, initDefaults.shareSkilled));
    decayLow  = max(0, min(0.999, initDefaults.wealthDecayLowSkill));
    decayHigh = max(0, min(0.999, initDefaults.wealthDecayHighSkill));

    % ------------------------------------------------------------------
    % Skill distribution: enforce 20% skilled (type 2 by default)
    % ------------------------------------------------------------------
    skilledIdx = min(2, dims.S); % Use skill type 2 as "skilled" when available
    otherIdx   = setdiff(1:dims.S, skilledIdx);

    if isempty(otherIdx)
        skillProb = 1; % Only one skill type available
    else
        shareOther = (1 - shareSkilled);
        skillProb = zeros(1, dims.S);
        skillProb(skilledIdx) = shareSkilled;
        skillProb(otherIdx) = shareOther / numel(otherIdx);
    end

    skillCDF = cumsum(skillProb);
    skillCDF(end) = 1; % Guard against numerical drift

    % ------------------------------------------------------------------
    % Wealth distributions conditional on skill
    % ------------------------------------------------------------------
    if dims.Na < 1
        error('createInitialDistribution:InvalidWealthGrid', ...
              'dims.Na must be at least 1.');
    end

    wealthCDF = cell(dims.S, 1);
    wealthIdx = 0:(dims.Na - 1);

    for s = 1:dims.S
        % Use two decay rates: one for skilled, one for the remaining skill types
        if s == skilledIdx
            decay = decayHigh;
        else
            decay = decayLow;
        end

        baseWeights = (1 - decay) .^ wealthIdx;

        % Guard against degenerate cases (all zero mass)
        if all(baseWeights == 0)
            baseWeights(1) = 1;
        end

        wealthProb = baseWeights / sum(baseWeights);
        wealthCDF{s} = cumsum(wealthProb);
        wealthCDF{s}(end) = 1; % Ensure full coverage of the unit interval
    end

    % ------------------------------------------------------------------
    % Preallocate struct array and populate agent attributes
    % ------------------------------------------------------------------
    m0 = repmat(struct( ...
        'skill',    0, ...
        'state',    0, ...
        'wealth',   0, ...
        'location', 0, ...
        'network',  0), numAgents, 1);

    drawsSkill = rand(numAgents, 1);

    for i = 1:numAgents
        m0(i).skill = find(drawsSkill(i) <= skillCDF, 1, 'first');

        wealthDraw = rand();
        m0(i).wealth = find(wealthDraw <= wealthCDF{m0(i).skill}, 1, 'first');

        m0(i).state    = 1;  % Employment × ψ state (kept deterministic for now)
        m0(i).location = 1;  % All start in Venezuela (location 1)
        m0(i).network  = 1;  % All start network-affiliated
    end

end
