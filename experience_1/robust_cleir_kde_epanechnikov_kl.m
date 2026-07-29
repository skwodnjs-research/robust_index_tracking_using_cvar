SELECTED = 100;

%% ============================================================
% 데이터 불러오기
% =============================================================

index = ["sp500", "russell2000", "kospi"];
INDEX = ["SP500", "Russell2000", "KOSPI"];

selected = 1;

index = index(selected);
INDEX = INDEX(selected);

% 파일 경로
stockFile = fullfile("data", "data_" + index + "_stocks.csv");
indexFile = fullfile("data", "data_" + index + "_index.csv");

% CSV 파일 불러오기
stockTable = readtable(stockFile, "VariableNamingRule", "preserve");
indexTable = readtable(indexFile, "VariableNamingRule", "preserve");

% 첫 번째 열을 날짜로 설정
stockTable.Date = datetime(stockTable.Date);
indexTable.Date = datetime(indexTable.Date);

% 데이터 불러오기 완료
stockPrices = table2timetable(stockTable, "RowTimes", "Date");
indexPrice = table2timetable(indexTable, "RowTimes", "Date");

%% ============================================================
% 데이터 전처리
% =============================================================

stockPrices = sortrows(stockPrices);
indexPrice = sortrows(indexPrice);

stockData = stockPrices{:,:};
stockData(isinf(stockData)) = NaN;
stockPrices{:,:} = stockData;

indexData = indexPrice{:,:};
indexData(isinf(indexData)) = NaN;
indexPrice{:,:} = indexData;

% 1. 날짜가 중복된 행이 있으면 첫 번째 행만 남기고 나머지는 삭제
[~, stockUniqueIdx] = unique(stockPrices.Date, "stable");
stockPrices = stockPrices(stockUniqueIdx, :);

[~, indexUniqueIdx] = unique(indexPrice.Date, "stable");
indexPrice = indexPrice(indexUniqueIdx, :);

stockPrices = sortrows(stockPrices);
indexPrice = sortrows(indexPrice);

% 2. return 데이터 생성
stockReturns = stockPrices(2:end, :);
stockReturns{:,:} = stockPrices{2:end, :} ./ stockPrices{1:end-1, :} - 1;

indexReturn = indexPrice(2:end, :);
indexReturn{:,:} = indexPrice{2:end, :} ./ indexPrice{1:end-1, :} - 1;

stockReturnData = stockReturns{:,:};
stockReturnData(isinf(stockReturnData)) = NaN;
stockReturns{:,:} = stockReturnData;

indexReturnData = indexReturn{:,:};
indexReturnData(isinf(indexReturnData)) = NaN;
indexReturn{:,:} = indexReturnData;

% 3. 벤치마크 수익률이 NaN인 날짜 제거
validIndexRows = ~isnan(indexReturn{:,1});
indexReturn = indexReturn(validIndexRows, :);

% 4. 모든 종목의 수익률이 NaN인 날짜 제거
allStockNaN = all(isnan(stockReturns{:,:}), 2);
stockReturns = stockReturns(~allStockNaN, :);

% 5. 주식 데이터와 벤치마크 데이터가 모두 유효한 날짜만을 사용
commonReturnDates = intersect(stockReturns.Date, indexReturn.Date);

stockReturns = stockReturns(ismember(stockReturns.Date, commonReturnDates), :);
indexReturn = indexReturn(ismember(indexReturn.Date, commonReturnDates), :);

stockReturns = sortrows(stockReturns);
indexReturn = sortrows(indexReturn);

% 결과 출력
datesMatched = isequal(stockReturns.Date, indexReturn.Date);

fprintf("Loaded data\n");
fprintf("Stock returns      : %d x %d\n", height(stockReturns), width(stockReturns));
fprintf("Benchmark returns  : %d x %d\n", height(indexReturn), width(indexReturn));
fprintf("Dates matched      : %s\n", string(datesMatched));

%% ============================================================
% Parameter 설정
% =============================================================

alpha = 0.95;
h = 1;
tau = 0.01;
s = 1.5;


%% ============================================================
% Rolling-window optimization
% =============================================================

dates = stockReturns.Date;
firstRebalanceYear = min(year(dates)) + 2;
lastRebalanceYear  = min(max(year(dates)), 2025);
rebalanceYears = firstRebalanceYear:lastRebalanceYear;

start = tic;

stockNames = string(stockReturns.Properties.VariableNames);

weightHistory = array2timetable( ...
    zeros(0, numel(stockNames)), ...
    "RowTimes", datetime.empty(0,1), ...
    "VariableNames", stockNames ...
);

optimizationHistory = array2timetable( ...
    zeros(0,3), ...
    "RowTimes", datetime.empty(0,1), ...
    "VariableNames", [ ...
        "objective_value", ...
        "exitflag", ...
        "solve_time_seconds" ...
    ] ...
);

backtestReturns = array2timetable( ...
    zeros(0,2), ...
    "RowTimes", datetime.empty(0,1), ...
    "VariableNames", [ ...
        "benchmark", ...
        "portfolio" ...
    ] ...
);

for rebalanceYear = rebalanceYears

    trainingYear = rebalanceYear - 1;
    trainingMask = year(dates) == trainingYear;
    trainingReturns = stockReturns(trainingMask,:);
    trainingBenchmark = indexReturn(trainingMask,:);

    if height(trainingReturns) == 0
        fprintf("%d skipped: no training data for %d\n", rebalanceYear, trainingYear);
        continue;
    end
    
    holdingMask = year(stockReturns.Properties.RowTimes) == rebalanceYear;
    holdingDates = stockReturns.Properties.RowTimes(holdingMask);

    if isempty(holdingDates)
        fprintf("%d skipped: no holding data\n", rebalanceYear);
        continue;
    end
    
    rebalanceDate = holdingDates(1);
    
    % 투자 종목 설정
    validStocks = mean(~isnan(trainingReturns{:,:}), 1) >= 0.95;
    validColumns = trainingReturns.Properties.VariableNames(validStocks);

    if isempty(validColumns)
        fprintf("%s skipped: no valid stocks\n", string(rebalanceDate));
        continue;
    end

    nSelected = min(SELECTED, numel(validColumns));
    selectedColumns = validColumns(1:nSelected);

    % 학습 데이터 설정
    trainingReturns = trainingReturns(:, selectedColumns);
    trainingReturns{:,:} = fillmissing(trainingReturns{:,:}, "constant", 0);

    commonDates = intersect(trainingReturns.Properties.RowTimes, trainingBenchmark.Properties.RowTimes);

    trainingReturns = trainingReturns(commonDates,:);
    trainingBenchmark = trainingBenchmark(commonDates,:);

    if height(trainingReturns) == 0
        fprintf("%s skipped: no matched training dates\n", string(rebalanceDate));
        continue;
    end

    t = tic;

    % 최적화
    R = trainingReturns{:,:};
    Y = trainingBenchmark{:,1};
    xi = [Y, R];
    
    [wOptimal, objectiveValue, exitflag] = optimizePortfolio(xi, alpha, tau, h, s);
    
    if exitflag <= 0
        fprintf("Rebalance %d optimization failed\n", rebalanceYear);
        continue;
    end
    
    fprintf("Rebalance %d finished (%.3f s)\n", rebalanceYear, toc(t));

    % 저장

    solveTimeSeconds = toc(t);

    % ---------------------------------------------------------
    % Weight history
    % ---------------------------------------------------------

    fullWeights = zeros(1, numel(stockNames));

    [isSelected, selectedLocations] = ismember( ...
        selectedColumns, ...
        stockNames ...
    );

    fullWeights(selectedLocations(isSelected)) = ...
        wOptimal(isSelected);

    newWeightHistory = array2timetable( ...
        fullWeights, ...
        "RowTimes", rebalanceDate, ...
        "VariableNames", stockNames ...
    );

    weightHistory = [
        weightHistory
        newWeightHistory
    ];

    % ---------------------------------------------------------
    % Optimization history
    % ---------------------------------------------------------

    newOptimizationHistory = array2timetable( ...
        [objectiveValue, exitflag, solveTimeSeconds], ...
        "RowTimes", rebalanceDate, ...
        "VariableNames", [ ...
            "objective_value", ...
            "exitflag", ...
            "solve_time_seconds" ...
        ] ...
    );

    optimizationHistory = [
        optimizationHistory
        newOptimizationHistory
    ];

    % ---------------------------------------------------------
    % Backtest returns
    % ---------------------------------------------------------

    holdingMask = ...
        year(stockReturns.Properties.RowTimes) == rebalanceYear;

    holdingReturns = ...
        stockReturns(holdingMask, selectedColumns);

    holdingBenchmark = indexReturn( ...
        year(indexReturn.Properties.RowTimes) == rebalanceYear, ...
        : ...
    );

    commonHoldingDates = intersect( ...
        holdingReturns.Properties.RowTimes, ...
        holdingBenchmark.Properties.RowTimes ...
    );

    if isempty(commonHoldingDates)
        fprintf( ...
            "%d skipped: no matched holding dates\n", ...
            rebalanceYear ...
        );
        continue;
    end

    holdingReturns = ...
        holdingReturns(commonHoldingDates, :);

    holdingBenchmark = ...
        holdingBenchmark(commonHoldingDates, :);

    holdingReturns{:,:} = fillmissing( ...
        holdingReturns{:,:}, ...
        "constant", ...
        0 ...
    );

    portfolioReturns = ...
        holdingReturns{:,:} * wOptimal;

    benchmarkReturns = ...
        holdingBenchmark{:,1};

    newBacktestReturns = array2timetable( ...
        [benchmarkReturns, portfolioReturns], ...
        "RowTimes", commonHoldingDates, ...
        "VariableNames", [ ...
            "benchmark", ...
            "portfolio" ...
        ] ...
    );

    backtestReturns = [
        backtestReturns
        newBacktestReturns
    ];

end

fprintf("finished (%.3f s)\n", toc(start));

%% ============================================================
% 저장
% =============================================================

if ~isfolder("data")
    mkdir("data");
end

weightFile = fullfile( ...
    "data", ...
    "data_" + index + ...
    "_robust_cleir_kde_epanechnikov_kl_weights.csv" ...
);

optimizationFile = fullfile( ...
    "data", ...
    "data_" + index + ...
    "_robust_cleir_kde_epanechnikov_kl_optimization_history.csv" ...
);

returnsFile = fullfile( ...
    "data", ...
    "data_" + index + ...
    "_robust_cleir_kde_epanechnikov_kl_returns.csv" ...
);

writetimetable( ...
    weightHistory, ...
    weightFile ...
);

writetimetable( ...
    optimizationHistory, ...
    optimizationFile ...
);

writetimetable( ...
    backtestReturns, ...
    returnsFile ...
);

%% ============================================================
% 함수 선언
% =============================================================

function [wOptimal, objectiveValue, exitflag] = optimizePortfolio( ...
    xi, alpha, tau, h, s)

    m = size(xi(:,2:end),2);
    
    % 최적화 변수 : [w; a; eta; u]
    % 초기값 설정
    w0   = ones(m,1)/m;
    gamma0   = 0;
    eta0 = 0;
    u0   = 1;
    
    v0 = abs(w0);
    x0 = [w0; gamma0; eta0; u0; v0];
    
    % 등식 제약조건: sum(w) = 1
    Aeq = [ones(1,m), 0, 0, 0, zeros(1,m)];
    beq = 1;
    A = [
        eye(m), zeros(m,3), -eye(m);
        -eye(m), zeros(m,3), -eye(m);
        zeros(1,m+3), ones(1,m)
        ];

    b = [
        zeros(2*m,1);
        s
        ];
    
    % 변수 범위
    lb = [-Inf(m,1); -Inf; -Inf; 1e-8; zeros(m,1)];
    ub = [ Inf(m,1);  Inf;  Inf; Inf; Inf(m,1)];
    
    % 목적함수
    objective = @(x) robustObjective(x(1:m+3), xi, alpha, tau, h);
    
    % fmincon 설정
    options = optimoptions("fmincon", ...
        "Algorithm", "interior-point", ...
        "Display", "none", ...
        "MaxIterations", 1000, ...
        "MaxFunctionEvaluations", 1e5);
    
    % 최적화
    [xOptimal, objectiveValue, exitflag] = fmincon( ...
        objective, ...
        x0, ...
        A, b, ...
        Aeq, beq, ...
        lb, ub, ...
        [], ...
        options);
    
    % 포트폴리오 비중만 추출
    wOptimal = xOptimal(1:m);
end

function value = Psi(u, h)
    value = zeros(size(u));

    idx1 = (u < -h);
    idx2 = (-h <= u) & (u <= h);
    idx3 = (u > h);

    value(idx1) = 0;
    value(idx2) = -u(idx2).^4 ./ (16 .* h.^3) + 3 .* u(idx2).^2 ./ (8 .* h) + 0.5 .* u(idx2) + 3 .* h ./ 16;
    value(idx3) = u(idx3);
end

function value = phiStar(q)
    value = exp(q) - 1;
end

function value = robustObjective(x, xi, alpha, tau, h)
    R = xi(:,2:end);

    n = size(R,1);
    m = size(R,2);

    w = x(1:m);
    wTilde = [-1; w];
    gamma = x(m+1);
    eta = x(m+2);
    u = x(m+3);

    loss = -xi * wTilde;
    psiValue = Psi(-loss-gamma, h);

    q = (psiValue./(1-alpha) - eta) ./ u;
    value = gamma + u*tau + eta + (u/n)*sum(phiStar(q));
end