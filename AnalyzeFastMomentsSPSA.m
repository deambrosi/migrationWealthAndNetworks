function results = AnalyzeFastMomentsSPSA(userOpt)
% ANALYZEFASTMOMENTSSPSA  Explore sensitivity of fastSimulatedMoments via SPSA.
%
%   RESULTS = ANALYZEFASTMOMENTSSPSA(USEROPT) runs the fastSimulatedMoments
%   simulator at the baseline parameter vector and at 2*B stochastic
%   perturbations defined by the Simultaneous Perturbation Stochastic
%   Approximation (SPSA) scheme. The routine perturbs the 36 calibrated
%   parameters described in the project notes, computes the 211 simulated
%   moments of interest at each draw, estimates the SPSA gradient, and
%   summarises the co-movement between parameters and moments with a
%   correlation heat map.
%
%   USEROPT (struct, optional) fields:
%       .B                Number of Rademacher draws for SPSA averaging (default 20).
%       .cFraction        Relative perturbation size as a fraction of range (default 0.05).
%       .displayLevel     0 = silent, 1 = iteration summary (default 1).
%       .rngSeed          Seed for reproducibility (default: rng('shuffle')).
%       .sim              Struct passed to fastSimulatedMoments (merged with defaults
%                         below). Empty or missing fields keep defaults.
%
%   Default simulation overrides passed to fastSimulatedMoments:
%       A1_start = 2, A1_end = 1, B1_start = 1.2, B1_end = 0.01,
%       TransitionPeriods = 10, years = 20.
%
%   OUTPUT STRUCT RESULTS fields:
%       .theta0           Baseline stacked parameter vector (36×1).
%       .thetaBounds      Struct with .lower and .upper bounds (36×1 each).
%       .thetaSamples     Matrix of parameter draws used in simulations (40×36 when B=20).
%       .momentSamples    Matrix of moment vectors for each draw (40×211 when B=20).
%       .grad             SPSA gradient estimate of moments wrt parameters (211×36).
%       .corrMatrix       Correlation matrix between parameters and moments (36×211).
%       .paramNames       Cell array with parameter labels (1×36).
%       .momentNames      Cell array with moment labels (1×211).
%       .figHandle        Handle to generated heat map figure.
%
%   The procedure echoes progress for each of the 2*B simulations so that
%   long runs can be monitored from the MATLAB/Octave console.

    if nargin < 1 || isempty(userOpt)
        userOpt = struct();
    end

    dims = setDimensionParam();
    baseParams = SetParameters(dims);

    optDefaults.B            = 20;
    optDefaults.cFraction    = 0.05;
    optDefaults.displayLevel = 1;
    optDefaults.sim          = struct();

    opt = mergeStructs(optDefaults, userOpt);

    if ~isfield(opt, 'rngSeed') || isempty(opt.rngSeed)
        rng('shuffle');
    else
        rng(opt.rngSeed);
    end

    if ~isstruct(opt.sim)
        error('AnalyzeFastMomentsSPSA:InvalidSimOpt', 'opt.sim must be a struct.');
    end

    simOpt = mergeStructs(struct('A1_start', 2, 'A1_end', 1, ...
        'B1_start', 1.2, 'B1_end', 0.01, 'TransitionPeriods', 10, ...
        'years', 20), opt.sim);

    [theta0, paramNames] = packTheta(baseParams);
    bounds = parameterBounds(baseParams);

    numParams  = numel(theta0);
    numMoments = 211;

    cStep = opt.cFraction .* (bounds.upper - bounds.lower);
    cStep = max(cStep, 1e-4);

    B = opt.B;
    if ~isscalar(B) || B <= 0 || B ~= round(B)
        error('AnalyzeFastMomentsSPSA:InvalidB', 'opt.B must be a positive integer.');
    end

    totalRuns = 2 * B;
    thetaSamples  = nan(totalRuns, numParams);
    momentSamples = nan(totalRuns, numMoments);
    deltas        = nan(B, numParams);
    gradAccum     = zeros(numMoments, numParams);

    runIdx = 0;
    momentNames = {};
    for b = 1:B
        Delta = 2 * (rand(1, numParams) > 0.5) - 1;
        deltas(b, :) = Delta;

        thetaPlus  = projectToBounds(theta0 + cStep .* Delta', bounds);
        thetaMinus = projectToBounds(theta0 - cStep .* Delta', bounds);

        if opt.displayLevel >= 1
            fprintf('[%s] SPSA draw %02d/%02d – positive perturbation.\n', ...
                datestr(now, 'HH:MM:SS'), b, B);
        end
        runIdx = runIdx + 1;
        [momPlus, ~, namesPlus] = evaluateTheta(thetaPlus, baseParams, simOpt);
        thetaSamples(runIdx, :)  = thetaPlus(:)';
        momentSamples(runIdx, :) = momPlus(:)';
        if isempty(momentNames)
            momentNames = namesPlus;
        end

        if opt.displayLevel >= 1
            fprintf('[%s] SPSA draw %02d/%02d – negative perturbation.\n', ...
                datestr(now, 'HH:MM:SS'), b, B);
        end
        runIdx = runIdx + 1;
        [momMinus, ~, namesMinus] = evaluateTheta(thetaMinus, baseParams, simOpt);
        thetaSamples(runIdx, :)  = thetaMinus(:)';
        momentSamples(runIdx, :) = momMinus(:)';
        if isempty(momentNames)
            momentNames = namesMinus;
        end

        diffMom = momPlus - momMinus;
        diffMom(isnan(diffMom)) = 0;
        denom = 2 .* (cStep(:)' .* Delta);
        gradAccum = gradAccum + diffMom ./ denom;
    end

    gradEstimate = gradAccum ./ B;

    corrMatrix = corr(thetaSamples, momentSamples, 'Rows', 'pairwise');

    columnGroups = buildColumnGroups();
    rowGroups = buildRowGroups();

    figHandle = plotCorrelationHeatmap(corrMatrix, paramNames, momentNames, ...
        rowGroups, columnGroups);

    results = struct();
    results.theta0        = theta0;
    results.thetaBounds   = bounds;
    results.thetaSamples  = thetaSamples;
    results.momentSamples = momentSamples;
    results.grad          = gradEstimate;
    results.corrMatrix    = corrMatrix;
    results.paramNames    = paramNames;
    results.momentNames   = momentNames;
    results.deltas        = deltas;
    results.cStep         = cStep;
    results.figHandle     = figHandle;
    results.rowGroups     = rowGroups;
    results.columnGroups  = columnGroups;
    results.simOptions    = simOpt;
end

function s = mergeStructs(a, b)
    if nargin < 1 || isempty(a)
        a = struct();
    end
    if nargin < 2 || isempty(b)
        s = a;
        return;
    end
    s = a;
    fieldsB = fieldnames(b);
    for i = 1:numel(fieldsB)
        key = fieldsB{i};
        val = b.(key);
        if isstruct(val)
            if isfield(a, key) && isstruct(a.(key))
                s.(key) = mergeStructs(a.(key), val);
            else
                s.(key) = val;
            end
        else
            s.(key) = val;
        end
    end
end

function [theta, names] = packTheta(params)
    names = cell(1, 36);
    theta = nan(36, 1);
    idx = 0;

    for loc = 3:6
        idx = idx + 1;
        theta(idx) = params.A(loc);
        names{idx} = sprintf('A_%d', loc);
    end

    for loc = 3:6
        idx = idx + 1;
        theta(idx) = params.B(loc);
        names{idx} = sprintf('B_%d', loc);
    end

    for loc = 2:6
        idx = idx + 1;
        theta(idx) = params.f(1, loc, 1);
        names{idx} = sprintf('f0_%d', loc);
    end

    for loc = 2:6
        idx = idx + 1;
        theta(idx) = params.f(1, loc, 2);
        names{idx} = sprintf('f1_%d', loc);
    end

    for loc = 2:6
        idx = idx + 1;
        theta(idx) = params.up_psi(loc);
        names{idx} = sprintf('xi_%d', loc);
    end

    for loc = 2:6
        idx = idx + 1;
        theta(idx) = params.ttau(1, loc);
        names{idx} = sprintf('hat_tau_%d', loc);
    end

    for loc = 2:6
        idx = idx + 1;
        theta(idx) = params.ttau(loc, 1);
        names{idx} = sprintf('tilde_tau_%d', loc);
    end

    idx = idx + 1;
    theta(idx) = params.phiH;
    names{idx} = 'phiH';

    idx = idx + 1;
    theta(idx) = params.aalpha;
    names{idx} = 'alpha';

    idx = idx + 1;
    theta(idx) = params.ggamma;
    names{idx} = 'gamma';
end

function bounds = parameterBounds(params)
    lower = nan(36, 1);
    upper = nan(36, 1);

    idx = 0;
    for ~ = 1:4
        idx = idx + 1;
        lower(idx) = 1;
        upper(idx) = 5;
    end

    for ~ = 1:4
        idx = idx + 1;
        lower(idx) = 1;
        upper(idx) = 3;
    end

    for ~ = 1:5
        idx = idx + 1;
        lower(idx) = 0.05;
        upper(idx) = 0.5;
    end

    for ~ = 1:5
        idx = idx + 1;
        lower(idx) = 0.40;
        upper(idx) = 0.8;
    end

    for ~ = 1:5
        idx = idx + 1;
        lower(idx) = 0.05;
        upper(idx) = 0.25;
    end

    for ~ = 1:10
        idx = idx + 1;
        lower(idx) = 0.5;
        upper(idx) = 20;
    end

    idx = idx + 1;
    lower(idx) = 0.2;
    upper(idx) = 2;

    idx = idx + 1;
    lower(idx) = 0.4;
    upper(idx) = 0.9;

    idx = idx + 1;
    lower(idx) = 0.4;
    upper(idx) = 3;

    bounds.lower = lower;
    bounds.upper = upper;
end

function thetaProj = projectToBounds(theta, bounds)
    thetaProj = min(max(theta, bounds.lower), bounds.upper);
end

function [momVec, out, momentNames] = evaluateTheta(theta, baseParams, simOpt)
    overrides = thetaToOverrides(theta, baseParams);
    out = fastSimulatedMoments(overrides, simOpt);
    [momVec, momentNames] = collectMoments(out);
end

function overrides = thetaToOverrides(theta, baseParams)
    overrides = struct();
    idx = 0;

    newA = baseParams.A;
    for loc = 3:6
        idx = idx + 1;
        newA(loc) = theta(idx);
    end
    overrides.A = newA;

    newB = baseParams.B;
    for loc = 3:6
        idx = idx + 1;
        newB(loc) = theta(idx);
    end
    overrides.B = newB;

    newF = baseParams.f;
    for loc = 2:6
        idx = idx + 1;
        newF(1, loc, 1) = theta(idx);
    end
    for loc = 2:6
        idx = idx + 1;
        newF(1, loc, 2) = theta(idx);
    end
    overrides.f = newF;

    newUp = baseParams.up_psi;
    for loc = 2:6
        idx = idx + 1;
        newUp(loc) = theta(idx);
    end
    overrides.up_psi = newUp;

    newTau = baseParams.ttau;
    for loc = 2:6
        idx = idx + 1;
        newTau(1, loc) = theta(idx);
    end
    for loc = 2:6
        idx = idx + 1;
        newTau(loc, 1) = theta(idx);
    end
    overrides.ttau = newTau;

    idx = idx + 1;
    overrides.phiH = theta(idx);

    idx = idx + 1;
    overrides.aalpha = theta(idx);

    idx = idx + 1;
    overrides.ggamma = theta(idx);
end

function [moments, names] = collectMoments(out)
    cross = out.crossSection;
    flows = out.flows;
    cohorts = out.cohorts;
    M_total = out.M_total;
    settings = out.settings;

    numYearsCross = min(7, size(cross.avgIncome, 2));
    targetYears = 1:numYearsCross;
    calendarYears = 2015 + targetYears;

    moments = [];
    names = {};

    locsIncome = [2, 3, 4, 6];
    for loc = locsIncome
        for yIdx = 1:numel(targetYears)
            val = cross.avgIncome(loc, targetYears(yIdx));
            moments(end+1, 1) = sanitizeMoment(val); %#ok<AGROW>
            names{end+1} = sprintf('avgIncome_L%d_Y%d', loc, calendarYears(yIdx)); %#ok<AGROW>
        end
    end

    for loc = locsIncome
        for yIdx = 1:numel(targetYears)
            val = cross.unemployment(loc, targetYears(yIdx));
            moments(end+1, 1) = sanitizeMoment(val);
            names{end+1} = sprintf('unemp_L%d_Y%d', loc, calendarYears(yIdx));
        end
    end

    locsHelpFull = [2, 3, 6];
    for loc = locsHelpFull
        for yIdx = 1:numel(targetYears)
            val = flows.shareHelp(loc, targetYears(yIdx));
            moments(end+1, 1) = sanitizeMoment(val);
            names{end+1} = sprintf('shareHelp_L%d_Y%d', loc, calendarYears(yIdx));
        end
    end
    loc4HelpYears = 1:min(4, size(flows.shareHelp, 2));
    for yIdx = 1:numel(loc4HelpYears)
        year = loc4HelpYears(yIdx);
        val = flows.shareHelp(4, year);
        moments(end+1, 1) = sanitizeMoment(val);
        names{end+1} = sprintf('shareHelp_L4_Y%d', 2015 + year);
    end

    locsDirectFull = [2, 3, 6];
    for loc = locsDirectFull
        for yIdx = 1:numel(targetYears)
            val = flows.shareDirectVzla(loc, targetYears(yIdx));
            moments(end+1, 1) = sanitizeMoment(val);
            names{end+1} = sprintf('shareDirect_L%d_Y%d', loc, calendarYears(yIdx));
        end
    end
    loc4DirectYears = 1:min(4, size(flows.shareDirectVzla, 2));
    for yIdx = 1:numel(loc4DirectYears)
        year = loc4DirectYears(yIdx);
        val = flows.shareDirectVzla(4, year);
        moments(end+1, 1) = sanitizeMoment(val);
        names{end+1} = sprintf('shareDirect_L4_Y%d', 2015 + year);
    end

    cohortSpecs = {
        cohorts.loc2_year7, 2, 7;
        cohorts.loc3_year7, 3, 7;
        cohorts.loc4_year4, 4, 4;
        cohorts.loc5_year3, 5, 3;
        cohorts.loc5_year7_all, 5, 7;
        cohorts.loc6_year7_all, 6, 7;
        };

    for c = 1:size(cohortSpecs, 1)
        summary = cohortSpecs{c, 1};
        loc = cohortSpecs{c, 2};
        surveyYear = cohortSpecs{c, 3};
        arrYears = summary.cohortYears;
        for j = 1:numel(arrYears)
            val = summary.avgIncome(j);
            moments(end+1, 1) = sanitizeMoment(val);
            names{end+1} = sprintf('cohortIncome_L%d_arr%d_obs%d', loc, 2015 + arrYears(j), 2015 + surveyYear);
        end
    end

    for c = 1:size(cohortSpecs, 1)
        summary = cohortSpecs{c, 1};
        loc = cohortSpecs{c, 2};
        surveyYear = cohortSpecs{c, 3};
        arrYears = summary.cohortYears;
        for j = 1:numel(arrYears)
            val = summary.unempRate(j);
            moments(end+1, 1) = sanitizeMoment(val);
            names{end+1} = sprintf('cohortUnemp_L%d_arr%d_obs%d', loc, 2015 + arrYears(j), 2015 + surveyYear);
        end
    end

    quartersPerYear = 4;
    horizon = min(settings.T, max(targetYears) * quartersPerYear);
    yearQuarters = quartersPerYear:quartersPerYear:horizon;
    stocksLocs = 2:6;
    for loc = stocksLocs
        for yIdx = 1:numel(yearQuarters)
            q = yearQuarters(yIdx);
            val = M_total(loc, q);
            moments(end+1, 1) = sanitizeMoment(val);
            names{end+1} = sprintf('stock_L%d_Y%d', loc, calendarYears(yIdx));
        end
    end

    if numel(moments) ~= 211
        error('collectMoments:MomentCountMismatch', ...
            'Expected 211 moments, obtained %d.', numel(moments));
    end
end

function val = sanitizeMoment(x)
    if isempty(x) || isnan(x)
        val = 0;
    else
        val = x;
    end
end

function rowGroups = buildRowGroups()
    rowGroups.names = {'A', 'B', 'f_0', 'f_1', 'xi', 'tau', 'phiH', 'alpha', 'gamma'};
    rowGroups.sizes = [4, 4, 5, 5, 5, 10, 1, 1, 1];
end

function columnGroups = buildColumnGroups()
    columnGroups.names = {'Avg income (cross)', 'Unemp (cross)', ...
        'Share with help', 'Share direct', 'Cohort income', 'Cohort unemp', ...
        'Migrant stocks'};
    columnGroups.sizes = [28, 28, 25, 25, 35, 35, 35];
end

function fig = plotCorrelationHeatmap(corrMatrix, rowLabels, columnLabels, rowGroups, columnGroups)
    fig = figure('Name', 'Parameter-Moment Correlations', 'Color', 'w');
    imagesc(corrMatrix, [-1, 1]);
    colormap(customDivergingMap(256));
    colorbar;
    axis tight;
    ax = gca;
    ax.YDir = 'normal';
    set(ax, 'YTick', 1:numel(rowLabels), 'YTickLabel', rowLabels, ...
        'XTick', 1:numel(columnLabels), 'XTickLabel', columnLabels, ...
        'TickLength', [0 0]);
    xtickangle(90);
    grid on;

    yBoundaries = cumsum(rowGroups.sizes) + 0.5;
    for y = 1:numel(yBoundaries)
        yline(yBoundaries(y), 'Color', [0.2 0.2 0.2], 'LineWidth', 0.75);
    end

    xBoundaries = cumsum(columnGroups.sizes) + 0.5;
    for x = 1:numel(xBoundaries)
        xline(xBoundaries(x), 'Color', [0.2 0.2 0.2], 'LineWidth', 0.75);
    end

    hold on;
    xCenters = cumsum(columnGroups.sizes) - columnGroups.sizes / 2;
    yCenters = cumsum(rowGroups.sizes) - rowGroups.sizes / 2;
    for g = 1:numel(columnGroups.names)
        text(xCenters(g), 0.4, columnGroups.names{g}, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontWeight', 'bold', 'Rotation', 90, 'Color', [0.2 0.2 0.2], ...
            'Clipping', 'off');
    end
    for g = 1:numel(rowGroups.names)
        text(0.5, yCenters(g), rowGroups.names{g}, ...
            'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', ...
            'FontWeight', 'bold', 'Color', [0.2 0.2 0.2], 'Clipping', 'off');
    end
    hold off;

    title('Correlation between parameters and simulated moments');
end

function cmap = customDivergingMap(n)
    if nargin < 1
        n = 256;
    end
    half = floor(n / 2);
    t = linspace(0, 1, half)';
    blue = [t * 0.2, t * 0.4 + 0.3, ones(half, 1)];
    red  = [ones(half, 1), flipud(t * 0.4 + 0.3), flipud(t * 0.2)];
    cmap = [blue; red];
    if size(cmap, 1) < n
        cmap(end+1, :) = [1, 0.9, 0.9]; %#ok<AGROW>
    end
end
