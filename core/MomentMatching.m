function results = MomentMatching(dataCsvPath, optSim, optMatch)
% MOMENTMATCHING  Estimate structural parameters via yearly moment matching.
%
%   RESULTS = MOMENTMATCHING(DATACSVPATH, OPTSIM, OPTMATCH) reads empirical
%   targets from DATACSVPATH, simulates the model with parameters restricted to
%   satisfy corridor-order constraints, and searches for the parameter vector
%   that minimizes the weighted sum of squared residuals between simulated and
%   empirical yearly moments. The function leaves the core simulator
%   (SimulatedMoments.m) untouched and operates purely as an outer estimation
%   loop.
%
%   INPUTS
%   ------
%   dataCsvPath : char/string
%       Path to a CSV file containing empirical targets in long format. See the
%       templates in data/moment_targets_example.csv or
%       data/moment_targets_empirical.csv for the required columns.

%
%   optSim : struct (optional)
%       Options passed directly to SimulatedMoments.m (e.g., optSim.fast,
%       optSim.seed). If omitted, defaults to an empty struct.
%
%   optMatch : struct (optional)
%       Controls the estimation routine. Supported fields include
%           .originIndex        Index of the origin country (default: 1).
%           .colombiaIndex      Index normalized to B=1 (default: 2).
%           .corridorOrder      Row vector listing locations along the Andean
%                               corridor (default: 1:dims.N).
%           .locationLookup     containers.Map or struct mapping location names
%                               (strings) to indices.
%           .skillLookup        containers.Map or struct mapping skill labels to
%                               indices (default: {'low','high'} -> 1,2).
%           .weightsColumn      Custom name of weight column in CSV (default:
%                               'weight').
%           .useLocalRefine     If true, run fmincon refinement after
%                               particleswarm (default: false).
%           .particleswarmOptions  Options object for particleswarm.
%           .fminconOptions        Options object for fmincon.
%           .yearlyWindow       Maximum # of post-burn years to match
%                               (default: 7 as requested).
%           .populationScale    Scalar to convert simulated shares into levels
%                               when reporting masses (default: settings.Nagents).
%
%   OUTPUT
%   ------
%   results : struct with fields
%       .theta0            Initial transformed parameter vector.
%       .thetaStar         Best-fit transformed vector.
%       .loss              Objective value at optimum.
%       .dataVector        Empirical moments (ordered as read from CSV).
%       .simVector         Simulated moments at optimum.
%       .weights           Observation weights.
%       .residuals         Simulated minus empirical moments.
%       .meta              Table describing each moment (location, year, etc.).
%       .overrides         Struct of parameter overrides passed to SetParameters.
%       .params            Full parameter struct evaluated at optimum.
%       .out               Output struct returned by SimulatedMoments at optimum.
%
%   The routine requires MATLAB's Global Optimization Toolbox for the
%   particleswarm solver. A local refinement via fmincon is optional.
%
%   NOTE: This file deliberately does not modify SimulatedMoments.m. All
%         additional post-processing occurs here.
%
% -------------------------------------------------------------------------

    if nargin < 1 || isempty(dataCsvPath)
        error('MomentMatching:MissingDataPath', ...
            'You must supply the path to a CSV file with empirical moments.');
    end
    if nargin < 2 || isempty(optSim)
        optSim = struct();
    end
    if nargin < 3 || isempty(optMatch)
        optMatch = struct();
    end

    if ~isfile(dataCsvPath)
        error('MomentMatching:DataNotFound', ...
            'Data CSV file not found: %s', dataCsvPath);
    end

    %% 1) Load baseline parameters -------------------------------------------
    dims        = setDimensionParam();
    baseParams  = SetParameters(dims);

    %% 2) Read empirical targets ---------------------------------------------
    dataTable = readtable(dataCsvPath, 'TextType', 'string');
    dataTable = standardizeMomentTable(dataTable);
    [dataVec, weights, meta, parsers] = buildDataVector(dataTable, dims, optMatch);

    %% 3) Parameter transforms and initial point -----------------------------
    transform = buildParameterTransform(baseParams, dims, optMatch);
    theta0    = transform.theta0;
    lb        = transform.lb;
    ub        = transform.ub;

    %% 4) Objective handle ----------------------------------------------------
    bestRecord.loss = inf;
    bestRecord.theta = theta0;
    bestRecord.overrides = transform.unpack(theta0);
    try
        bestRecord.out = SimulatedMoments(bestRecord.overrides, optSim);
        bestRecord.simVector = computeSimulatedVector(bestRecord.out, meta, parsers, optMatch);
        bestRecord.loss = computeLoss(bestRecord.simVector, dataVec, weights);
    catch ME
        warning('MomentMatching:InitialSimulationFailed', ...
            'Initial simulation failed: %s', ME.message);
        bestRecord.out = [];
        bestRecord.simVector = nan(size(dataVec));
        bestRecord.loss = inf;
    end

    function loss = objective(theta)
        overrides = transform.unpack(theta);
        try
            out = SimulatedMoments(overrides, optSim);
            simVec = computeSimulatedVector(out, meta, parsers, optMatch);
            loss = computeLoss(simVec, dataVec, weights);

            if loss < bestRecord.loss
                bestRecord.loss      = loss;
                bestRecord.theta     = theta;
                bestRecord.overrides = overrides;
                bestRecord.out       = out;
                bestRecord.simVector = simVec;
            end
        catch ME
            warning('MomentMatching:SimulationError', ...
                'Simulation failed for current theta: %s', ME.message);
            loss = realmax;
        end
    end

    %% 5) Global search via particle swarm -----------------------------------
    nTheta = numel(theta0);
    if ~isfield(optMatch, 'particleswarmOptions') || isempty(optMatch.particleswarmOptions)
        psOpts = optimoptions('particleswarm', ...
            'Display', 'iter', ...
            'SwarmSize', max(30, 5 * nTheta), ...
            'MaxIterations', 100, ...
            'UseVectorized', false);
    else
        psOpts = optMatch.particleswarmOptions;
    end

    [~, ~] = particleswarm(@objective, nTheta, lb, ub, psOpts);

    %% 6) Optional local refinement -----------------------------------------
    if isfield(optMatch, 'useLocalRefine') && optMatch.useLocalRefine
        if ~isfield(optMatch, 'fminconOptions') || isempty(optMatch.fminconOptions)
            fmOpts = optimoptions('fmincon', ...
                'Algorithm', 'interior-point', ...
                'Display', 'iter', ...
                'SpecifyObjectiveGradient', false);
        else
            fmOpts = optMatch.fminconOptions;
        end

        problem = struct();
        problem.objective = @objective;
        problem.x0        = bestRecord.theta;
        problem.lb        = lb;
        problem.ub        = ub;
        problem.options   = fmOpts;
        problem.solver    = 'fmincon';

        [~, ~] = fmincon(problem);
    end

    thetaStar = bestRecord.theta;
    lossStar  = bestRecord.loss;

    %% 7) Assemble outputs ---------------------------------------------------
    overridesStar = transform.unpack(thetaStar);
    paramsStar    = SetParameters(dims, overridesStar);

    results = struct();
    results.theta0     = theta0;
    results.thetaStar  = thetaStar;
    results.loss       = lossStar;
    results.dataVector = dataVec;
    results.weights    = weights;
    results.simVector  = bestRecord.simVector;
    results.residuals  = bestRecord.simVector - dataVec;
    results.meta       = meta;
    results.overrides  = overridesStar;
    results.params     = paramsStar;
    results.out        = bestRecord.out;
    results.transform  = transform;

end

%% -------------------------------------------------------------------------
function [dataVec, weights, meta, parsers] = buildDataVector(tbl, dims, optMatch)
% BUILDDATAVECTOR  Convert CSV table into numeric vector + metadata.

    requiredCols = {'moment_type', 'location', 'year', 'value'};
    for c = requiredCols
        if ~ismember(c{1}, tbl.Properties.VariableNames)
            error('MomentMatching:MissingColumn', ...
                'Required column "%s" missing from data CSV.', c{1});
        end
    end

    nObs = height(tbl);

    % Weight handling -------------------------------------------------------
    weightCol = 'weight';
    if isfield(optMatch, 'weightsColumn') && ~isempty(optMatch.weightsColumn)
        weightCol = optMatch.weightsColumn;
    end
    if ismember(weightCol, tbl.Properties.VariableNames)
        weights = toNumericColumn(tbl.(weightCol));
        if any(isnan(weights))
            error('MomentMatching:InvalidWeights', 'Weight column contains NaNs.');
        end
    else
        weights = ones(nObs, 1);
    end
    weights = double(weights(:));

    % Moment types ---------------------------------------------------------
    momentTypeRaw = lower(strtrim(tbl.moment_type));

    % Location parsing -----------------------------------------------------
    parsers.locationLookup = buildLocationLookup(dims, optMatch);
    locationRaw = tbl.location;
    locationIdx = zeros(nObs, 1);
    for i = 1:nObs
        locationIdx(i) = parseCategorical(locationRaw(i), parsers.locationLookup, ...
            sprintf('location (row %d)', i));
        if locationIdx(i) > dims.N
            error('MomentMatching:LocationOutOfRange', ...
                'Location index %g exceeds number of locations (%d).', ...
                locationIdx(i), dims.N);
        end
    end

    % Skill parsing --------------------------------------------------------
    parsers.skillLookup = buildSkillLookup(dims, optMatch);
    if ismember('skill', tbl.Properties.VariableNames)
        skillRaw = tbl.skill;
    else
        skillRaw = repmat({''}, nObs, 1);
    end
    skillIdx = nan(nObs, 1);
    for i = 1:nObs
        if isMissing(skillRaw(i))
            skillIdx(i) = NaN;
        else
            skillIdx(i) = parseCategorical(skillRaw(i), parsers.skillLookup, ...
                sprintf('skill (row %d)', i));
            if skillIdx(i) > dims.S
                error('MomentMatching:SkillOutOfRange', ...
                    'Skill index %g exceeds number of skills (%d).', ...
                    skillIdx(i), dims.S);
            end
        end
    end

    % Category parsing -----------------------------------------------------
    if ismember('category', tbl.Properties.VariableNames)
        category = lower(strtrim(string(tbl.category)));
    else
        category = strings(nObs, 1);
    end

    % Tenure years ---------------------------------------------------------
    if ismember('tenure_year', tbl.Properties.VariableNames)
        tenureYear = toNumericColumn(tbl.tenure_year);
    else
        tenureYear = NaN(nObs, 1);
    end

    % Years ----------------------------------------------------------------
    yearValues = toNumericColumn(tbl.year);
    yearIdx = zeros(nObs, 1);
    uniqueYears = unique(yearValues(~isnan(yearValues)));
    uniqueYears = sort(uniqueYears(:));
    for k = 1:numel(uniqueYears)
        yearIdx(yearValues == uniqueYears(k)) = k;
    end

    parsers.yearValues = uniqueYears;

    % Values ---------------------------------------------------------------
    values = toNumericColumn(tbl.value);

    meta = table(momentTypeRaw, locationIdx, skillIdx, yearValues, yearIdx, tenureYear, category, ...
        'VariableNames', {'moment_type', 'location', 'skill', 'year_value', 'year_idx', 'tenure_year', 'category'});

    dataVec = values(:);
end

%% -------------------------------------------------------------------------
function tbl = standardizeMomentTable(tbl)
% STANDARDIZEMOMENTTABLE  Clean variable names for reliable downstream access.

    varNames = tbl.Properties.VariableNames;
    newNames = cell(size(varNames));
    for i = 1:numel(varNames)
        raw = lower(strtrim(varNames{i}));
        raw = regexprep(raw, '[^a-z0-9]', '');
        switch raw
            case 'momenttype'
                newNames{i} = 'moment_type';
            case 'location'
                newNames{i} = 'location';
            case 'skill'
                newNames{i} = 'skill';
            case 'year'
                newNames{i} = 'year';
            case 'tenureyear'
                newNames{i} = 'tenure_year';
            case 'value'
                newNames{i} = 'value';
            case 'weight'
                newNames{i} = 'weight';
            case 'category'
                newNames{i} = 'category';
            otherwise
                newNames{i} = matlab.lang.makeValidName(varNames{i});
        end
    end
    tbl.Properties.VariableNames = matlab.lang.makeUniqueStrings(newNames);
end

%% -------------------------------------------------------------------------
function vec = toNumericColumn(col)
% TONUMERICCOLUMN  Convert assorted column types to double column vectors.

    if isnumeric(col)
        vec = double(col(:));
        return;
    end
    if islogical(col)
        vec = double(col(:));
        return;
    end
    if isstring(col)
        vec = str2double(col(:));
        return;
    end
    if iscell(col)
        vec = nan(numel(col), 1);
        for i = 1:numel(col)
            item = col{i};
            if isempty(item)
                vec(i) = NaN;
            elseif isstring(item)
                vec(i) = str2double(item);
            elseif ischar(item)
                vec(i) = str2double(strtrim(item));
            elseif isnumeric(item) || islogical(item)
                vec(i) = double(item);
            else
                vec(i) = str2double(string(item));
            end
        end
        return;
    end
    if iscategorical(col)
        vec = str2double(string(col(:)));
        return;
    end
    error('MomentMatching:UnsupportedColumnType', ...
        'Unsupported column type %s for numeric conversion.', class(col));
end

%% -------------------------------------------------------------------------

function lookup = buildLocationLookup(dims, optMatch)
% BUILDLOCATIONLOOKUP  Construct name -> index mapping for locations.

    if isfield(optMatch, 'locationLookup') && ~isempty(optMatch.locationLookup)
        lookup = normalizeLookup(optMatch.locationLookup);
        return;
    end

    lookup = containers.Map('KeyType', 'char', 'ValueType', 'double');

    % Canonical aliases for corridor locations. Each cell contains the labels
    % that should map to the corresponding location index when available in
    % the current dimensionality. This list intentionally skips "colombia" as
    % a standalone entry because the data distinguish border vs. rest.
    aliasSets = {
        {'venezuela', 'vzla', 'origin'};                % 1 = origin
        {'colombiaborder', 'colombia_border', 'border'};% 2 = Colombian border
        {'colombiarest', 'colombia_rest', 'colombia'};  % 3 = Rest of Colombia
        {'ecuador', 'ec'};                              % 4 = Ecuador
        {'peru', 'pe'};                                 % 5 = Peru
        {'chile', 'cl'};                                % 6 = Chile
        {'argentina', 'ar'};                            % 7 = Argentina
        {'usa', 'unitedstates', 'us'};                  % 8 = USA (fallback)
    };

    for i = 1:min(dims.N, numel(aliasSets))
        aliases = aliasSets{i};
        for a = 1:numel(aliases)
            key = lower(char(aliases{a}));
            if ~isKey(lookup, key)
                lookup(key) = i;
            end
        end
    end

    % Generic fallbacks (location1, location2, ...)
    for i = 1:dims.N
        key = lower(sprintf('location%d', i));
        lookup(key) = i;
    end
end

%% -------------------------------------------------------------------------
function lookup = buildSkillLookup(dims, optMatch)
% BUILDSKILLLOOKUP  Construct mapping for skill labels.

    if isfield(optMatch, 'skillLookup') && ~isempty(optMatch.skillLookup)
        lookup = normalizeLookup(optMatch.skillLookup);
        return;
    end

    lookup = containers.Map('KeyType', 'char', 'ValueType', 'double');

    defaultSkills = {'low', 'high'};
    for s = 1:min(numel(defaultSkills), dims.S)
        lookup(defaultSkills{s}) = s;
    end

    for s = 1:dims.S
        key = lower(sprintf('skill%d', s));
        lookup(key) = s;
    end
end

%% -------------------------------------------------------------------------
function lookup = normalizeLookup(inputLookup)
% NORMALIZELOOKUP  Convert struct or containers.Map to containers.Map.

    if isa(inputLookup, 'containers.Map')
        keyCell = keys(inputLookup);
        valueCell = values(inputLookup);
        lookup = containers.Map('KeyType', 'char', 'ValueType', 'double');
        for k = 1:numel(keyCell)
            lookup(lower(char(keyCell{k}))) = double(valueCell{k});
        end
        return;
    end
    if isstruct(inputLookup)
        fields = fieldnames(inputLookup);
        lookup = containers.Map('KeyType', 'char', 'ValueType', 'double');
        for f = 1:numel(fields)
            lookup(lower(fields{f})) = double(inputLookup.(fields{f}));
        end
        return;
    end
    error('MomentMatching:InvalidLookup', ...
        'Lookup must be a containers.Map or struct.');
end

%% -------------------------------------------------------------------------
function idx = parseCategorical(rawValue, lookup, label)
% PARSECATEGORICAL  Parse string/numeric value into index using lookup.

    if isMissing(rawValue)
        error('MomentMatching:MissingCategory', ...
            'Missing value encountered for %s.', label);
    end

    if isnumeric(rawValue)
        idx = double(rawValue);
    elseif isstring(rawValue) || ischar(rawValue)
        str = lower(strtrim(string(rawValue)));
        if strlength(str) == 0
            error('MomentMatching:MissingCategory', ...
                'Empty string encountered for %s.', label);
        end
        if isKey(lookup, char(str))
            idx = lookup(char(str));
        else
            numVal = str2double(str);
            if ~isnan(numVal)
                idx = numVal;
            else
                error('MomentMatching:UnknownCategory', ...
                    'Unknown label "%s" for %s.', str, label);
            end
        end
    else
        error('MomentMatching:UnsupportedCategoryType', ...
            'Unsupported data type for %s.', label);
    end

    if ~isscalar(idx) || idx < 1 || abs(idx - round(idx)) > 1e-8
        error('MomentMatching:InvalidIndex', ...
            'Parsed index for %s must be a positive integer.', label);
    end
    idx = round(idx);
end

%% -------------------------------------------------------------------------
function tf = isMissing(val)
% ISMISSING  Helper for empty skill/category detection.

    if isempty(val)
        tf = true;
    elseif isstring(val)
        tf = strlength(val) == 0;
    elseif ischar(val)
        tf = isempty(strtrim(val));
    elseif iscell(val)
        tf = all(cellfun(@isMissing, val));
    else
        tf = false;
    end
end

%% -------------------------------------------------------------------------
function loss = computeLoss(simVec, dataVec, weights)
% COMPUTELOSS  Weighted least-squares criterion.

    residuals = simVec(:) - dataVec(:);
    loss = sum((sqrt(weights(:)) .* residuals).^2);
end

%% -------------------------------------------------------------------------
function transform = buildParameterTransform(baseParams, dims, optMatch)
% BUILDPARAMETERTRANSFORM  Prepare mappings between theta and parameter structs.

    originIdx   = getFieldWithDefault(optMatch, 'originIndex', 1);
    colombiaIdx = getFieldWithDefault(optMatch, 'colombiaIndex', min(2, dims.N));
    corridor    = getFieldWithDefault(optMatch, 'corridorOrder', 1:dims.N);

    if numel(unique(corridor)) ~= dims.N
        error('MomentMatching:InvalidCorridor', ...
            'corridorOrder must list each location exactly once.');
    end

    freeBIdx = setdiff(1:dims.N, [originIdx, colombiaIdx]);

    % Baseline transforms ---------------------------------------------------
    thetaParts = {};
    lbParts = {};
    ubParts = {};

    % 1) Amenities B -------------------------------------------------------
    B_base = baseParams.B;
    B_base(originIdx)   = 0.1;  % enforce normalization
    B_base(colombiaIdx) = 1.0;

    logB = log(max(B_base(freeBIdx), eps));
    thetaParts{end+1} = logB;
    lbParts{end+1}    = log(1e-3) * ones(size(logB));
    ubParts{end+1}    = log(50)   * ones(size(logB));

    % 2) Migration cost segments ------------------------------------------
    segmentCount = dims.N - 1;
    if segmentCount < 1
        error('MomentMatching:TooFewLocations', ...
            'At least two locations are required.');
    end

    segmentsBase = zeros(segmentCount, 1);
    for k = 1:segmentCount
        i = corridor(k);
        j = corridor(k+1);
        cost_ij = baseParams.ttau(i, j);
        if cost_ij < 1
            cost_ij = 1;
        end
        segmentsBase(k) = max(cost_ij - 1, 1e-3);
    end
    logSegments = log(segmentsBase);

    thetaParts{end+1} = logSegments;
    lbParts{end+1}    = -5 * ones(segmentCount, 1);
    ubParts{end+1}    = 5  * ones(segmentCount, 1);

    sumSegments = sum(segmentsBase);
    sumSegments = min(sumSegments, 14 - 1e-3);
    zTotal = log(sumSegments / (14 - sumSegments));
    thetaParts{end+1} = zTotal;
    lbParts{end+1}    = -5;
    ubParts{end+1}    = 5;

    % 3) aalpha ------------------------------------------------------------
    aalpha = baseParams.aalpha;
    thetaParts{end+1} = logitScaled(aalpha, 0.2, 0.8);
    lbParts{end+1}    = -5;
    ubParts{end+1}    = 5;

    % 4) ggamma ------------------------------------------------------------
    ggamma = baseParams.ggamma;
    thetaParts{end+1} = log(max(ggamma, 1e-3));
    lbParts{end+1}    = log(1e-3) * ones(1,1);
    ubParts{end+1}    = log(1e3)  * ones(1,1);

    % 5) cchi --------------------------------------------------------------
    cchi = baseParams.cchi;
    thetaParts{end+1} = logitScaled(cchi, 0.02, 0.20);
    lbParts{end+1}    = -5;
    ubParts{end+1}    = 5;

    % 6) up_psi ------------------------------------------------------------
    upPsi = baseParams.up_psi;
    thetaParts{end+1} = logitScaled(upPsi, 0.02, 0.20);
    lbParts{end+1}    = -5 * ones(size(upPsi));
    ubParts{end+1}    = 5  * ones(size(upPsi));

    transform.theta0 = vertcat(thetaParts{:});
    transform.lb     = vertcat(lbParts{:});
    transform.ub     = vertcat(ubParts{:});

    transform.originIdx   = originIdx;
    transform.colombiaIdx = colombiaIdx;
    transform.freeBIdx    = freeBIdx;
    transform.corridor    = corridor;
    transform.segmentCount= segmentCount;

    transform.unpack = @(theta) unpackTheta(theta, transform);
end

%% -------------------------------------------------------------------------
function overrides = unpackTheta(theta, transform)
% UNPACKTHETA  Map transformed parameters into override struct.

    idx = 0;

    % B amenities ----------------------------------------------------------
    nB = numel(transform.freeBIdx);
    logB = theta(idx+1:idx+nB); idx = idx + nB;
    N = transform.segmentCount + 1;
    B = zeros(N, 1);
    B(transform.freeBIdx) = exp(logB);
    B(transform.originIdx)   = 0.1;
    B(transform.colombiaIdx) = 1.0;

    % Migration cost segments ---------------------------------------------
    nSeg = transform.segmentCount;
    zSegments = theta(idx+1:idx+nSeg); idx = idx + nSeg;
    zTotal    = theta(idx+1);          idx = idx + 1;

    segments = exp(zSegments);
    if sum(segments) <= 0
        segments = ones(size(segments));
    end
    segments = segments / sum(segments);
    totalMagnitude = 14 * sigmoid(zTotal);
    segments = segments * max(totalMagnitude, 0);

    corridor = transform.corridor;
    Ncorr = numel(corridor);
    ttau = zeros(Ncorr, Ncorr);
    cumulativeSegments = [0; cumsum(segments)];
    for a = 1:Ncorr
        ttau(corridor(a), corridor(a)) = 0;
        for b = a+1:Ncorr
            dist = cumulativeSegments(b) - cumulativeSegments(a);
            cost = 1 + dist;
            ttau(corridor(a), corridor(b)) = min(max(cost, 1), 15);
            ttau(corridor(b), corridor(a)) = min(max(cost, 1), 15);
        end
    end

    % aalpha ---------------------------------------------------------------
    zAalpha = theta(idx+1); idx = idx + 1;
    aalpha = scaleSigmoid(zAalpha, 0.2, 0.8);

    % ggamma ---------------------------------------------------------------
    zGgamma = theta(idx+1); idx = idx + 1;
    ggamma = exp(zGgamma);

    % cchi -----------------------------------------------------------------
    zCchi = theta(idx+1); idx = idx + 1;
    cchi = scaleSigmoid(zCchi, 0.02, 0.20);

    % up_psi ---------------------------------------------------------------
    zUpPsi = theta(idx+1:end);
    upPsi = scaleSigmoid(zUpPsi, 0.02, 0.20);

    overrides = struct();
    overrides.B       = B(:);
    overrides.ttau    = ttau;
    overrides.aalpha  = aalpha;
    overrides.ggamma  = ggamma;
    overrides.cchi    = cchi;
    overrides.up_psi  = upPsi(:);
end

%% -------------------------------------------------------------------------
function simVec = computeSimulatedVector(out, meta, parsers, optMatch)
% COMPUTESIMULATEDVECTOR  Construct simulated yearly moments matching meta.

    quartersPerYear = 4;
    maxYearsDefault = getFieldWithDefault(optMatch, 'yearlyWindow', 7);

    settings = out.settings;
    startQ   = settings.burn + 1;
    totalQuarters = settings.T - settings.burn;
    numYears = min(maxYearsDefault, floor(totalQuarters / quartersPerYear));

    requiredYears = max([0; meta.year_idx]);
    if numYears < requiredYears
        error('MomentMatching:InsufficientYears', ...
            'Simulation provides only %d usable years; data require year index %d.', ...
            numYears, requiredYears);
    end

    yearEndIdx = startQ + (1:numYears) * quartersPerYear - 1;
    yearEndIdx(yearEndIdx > settings.T) = settings.T;

    Nagents = settings.Nagents;
    if isfield(optMatch, 'populationScale') && ~isempty(optMatch.populationScale)
        populationScale = optMatch.populationScale;
    else
        populationScale = Nagents;
    end

    dims = out.dims;
    S = dims.S;
    N = dims.N;

    locTraj   = double(out.agentData.location);
    stateTraj = double(out.agentData.state);
    skillVec  = double(out.agentData.skill(:));

    % Location shares by skill and in total --------------------------------
    totalAgents = size(locTraj, 1);
    shareBySkill = zeros(S, N, numYears);
    shareTotal   = zeros(N, numYears);
    for y = 1:numYears
        q = yearEndIdx(y);
        loc_q = locTraj(:, q);
        for i = 1:N
            inLoc = (loc_q == i);
            countLoc = sum(inLoc);
            if countLoc == 0
                continue;
            end
            shareTotal(i, y) = countLoc / totalAgents;
            for s = 1:S
                shareBySkill(s, i, y) = sum(skillVec == s & inLoc) / totalAgents;
            end
        end
    end

    % Tenure calculations --------------------------------------------------
    T = size(locTraj, 2);
    arrivalQuarter = ones(size(locTraj));
    for q = 2:T
        moved = locTraj(:, q) ~= locTraj(:, q-1);
        arrivalQuarter(moved, q) = q;
        arrivalQuarter(~moved, q) = arrivalQuarter(~moved, q-1);
    end

    isUnemployed = stateTraj <= dims.k;

    maxTenureYears = 4;
    tenureCounts = zeros(N, maxTenureYears + 1);
    tenureUnemp  = zeros(N, maxTenureYears + 1);
    usableYears = min(numYears, maxTenureYears + 1);
    for y = 1:usableYears
        q = yearEndIdx(y);
        tenureQuarters = q - arrivalQuarter(:, q);
        tenureYearsZero = min(maxTenureYears, floor(tenureQuarters / quartersPerYear));
        for i = 1:N
            inLoc = (locTraj(:, q) == i);
            for d = 0:maxTenureYears
                idxTenure = d + 1;
                mask = inLoc & (tenureYearsZero == d);
                tenureCounts(i, idxTenure) = tenureCounts(i, idxTenure) + sum(mask);
                tenureUnemp(i, idxTenure)  = tenureUnemp(i, idxTenure)  + sum(isUnemployed(mask, q));
            end
        end
    end
    unempByTenure = zeros(N, maxTenureYears + 1);
    for i = 1:N
        for d = 0:maxTenureYears
            idxTenure = d + 1;
            if tenureCounts(i, idxTenure) > 0
                unempByTenure(i, idxTenure) = tenureUnemp(i, idxTenure) / tenureCounts(i, idxTenure);
            end
        end
    end

    % Arrivals per year ----------------------------------------------------
    helpUsed = false(size(locTraj));
    directFromVzla = false(size(locTraj));
    if isfield(out, 'flowLog') && ~isempty(out.flowLog)
        if isfield(out.flowLog, 'helpUsed')
            helpUsed = logical(out.flowLog.helpUsed);
        end
        if isfield(out.flowLog, 'directFromVzla')
            directFromVzla = logical(out.flowLog.directFromVzla);
        end
    end

    arrivalsHelp = zeros(N, numYears, 2);   % (:,:,1)=with help, (:,:,2)=without
    arrivalsOrigin = zeros(N, numYears, 2); % (:,:,1)=direct, (:,:,2)=stepping
    arrivalsTotal = zeros(N, numYears);

    qStart = max(2, startQ);
    qEnd   = min(settings.T, startQ + numYears * quartersPerYear - 1);
    for q = qStart:qEnd
        yearIdx = floor((q - startQ) / quartersPerYear) + 1;
        if yearIdx < 1 || yearIdx > numYears
            continue;
        end
        moved = locTraj(:, q) ~= locTraj(:, q-1);
        if ~any(moved)
            continue;
        end
        dest = locTraj(moved, q);
        helpFlags = helpUsed(moved, q);
        directFlags = directFromVzla(moved, q);
        for i = 1:N
            mask = (dest == i);
            if any(mask)
                countArrivals = sum(mask);
                arrivalsHelp(i, yearIdx, 1) = arrivalsHelp(i, yearIdx, 1) + sum(helpFlags(mask));
                arrivalsHelp(i, yearIdx, 2) = arrivalsHelp(i, yearIdx, 2) + sum(~helpFlags(mask));
                arrivalsOrigin(i, yearIdx, 1) = arrivalsOrigin(i, yearIdx, 1) + sum(directFlags(mask));
                arrivalsOrigin(i, yearIdx, 2) = arrivalsOrigin(i, yearIdx, 2) + sum(~directFlags(mask));
                arrivalsTotal(i, yearIdx)    = arrivalsTotal(i, yearIdx) + countArrivals;
            end
        end
    end

    % Assemble simulated vector -------------------------------------------
    nObs = height(meta);
    simVec = nan(nObs, 1);
    for obs = 1:nObs
        mType = meta.moment_type(obs);
        iLoc  = meta.location(obs);
        yearIdx = meta.year_idx(obs);
        skill = meta.skill(obs);
        tenureYear = meta.tenure_year(obs);

        switch mType
            case 'share_end_year'
                if yearIdx < 1 || yearIdx > numYears
                    error('MomentMatching:YearOutOfRange', ...
                        'Year index %d outside simulated range 1-%d.', yearIdx, numYears);
                end
                if isnan(skill)
                    simVec(obs) = shareTotal(iLoc, yearIdx);
                else
                    simVec(obs) = shareBySkill(skill, iLoc, yearIdx);
                end

            case 'unemployment_rate'
                if isnan(tenureYear)
                    error('MomentMatching:MissingTenure', ...
                        'Tenure year required for unemployment_rate moments.');
                end
                if tenureYear < 0 || tenureYear > maxTenureYears
                    error('MomentMatching:TenureRange', ...
                        'Tenure year %g outside supported range 0-%d.', tenureYear, maxTenureYears);
                end
                idxTenure = round(tenureYear) + 1;
                if abs(idxTenure - (tenureYear + 1)) > 1e-6
                    error('MomentMatching:TenureNonInteger', ...
                        'Tenure year must be integer-valued (received %g).', tenureYear);
                end
                simVec(obs) = unempByTenure(iLoc, idxTenure);

            case 'came_with_help'
                if yearIdx < 1 || yearIdx > numYears
                    error('MomentMatching:YearOutOfRange', ...
                        'Year index %d outside simulated range 1-%d.', yearIdx, numYears);
                end
                totalArr = arrivalsTotal(iLoc, yearIdx);
                if totalArr > 0
                    simVec(obs) = arrivalsHelp(iLoc, yearIdx, 1) / totalArr;
                else
                    simVec(obs) = 0;
                end

            case 'came_directly'
                if yearIdx < 1 || yearIdx > numYears
                    error('MomentMatching:YearOutOfRange', ...
                        'Year index %d outside simulated range 1-%d.', yearIdx, numYears);
                end
                totalArr = sum(arrivalsOrigin(iLoc, yearIdx, :), 3);
                totalArr = totalArr(1);
                if totalArr > 0
                    simVec(obs) = arrivalsOrigin(iLoc, yearIdx, 1) / totalArr;
                else
                    simVec(obs) = 0;
                end

            otherwise
                error('MomentMatching:UnknownMomentType', ...
                    'Unsupported moment_type "%s".', mType);
        end
    end
end

%% -------------------------------------------------------------------------
function val = getFieldWithDefault(s, field, defaultVal)
% GETFIELDWITHDEFAULT  Convenience accessor for optional struct fields.

    if isstruct(s) && isfield(s, field) && ~isempty(s.(field))
        val = s.(field);
    else
        val = defaultVal;
    end
end

%% -------------------------------------------------------------------------
function y = sigmoid(x)
% SIGMOID  Logistic sigmoid.

    y = 1 ./ (1 + exp(-x));
end

%% -------------------------------------------------------------------------
function z = logitScaled(x, lowerBound, upperBound)
% LOGITSCALED  Inverse of scaleSigmoid.

    range = upperBound - lowerBound;
    x = min(max(x, lowerBound + 1e-6), upperBound - 1e-6);
    xNorm = (x - lowerBound) / range;
    z = log(xNorm ./ (1 - xNorm));
end

%% -------------------------------------------------------------------------
function x = scaleSigmoid(z, lowerBound, upperBound)
% SCALESIGMOID  Map unconstrained z to [lowerBound, upperBound].

    range = upperBound - lowerBound;
    x = lowerBound + range * sigmoid(z);
end

