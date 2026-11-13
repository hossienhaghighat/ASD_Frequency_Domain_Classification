clc; clear;

%% ================= Paths & Parameters =================
baseFeaturePath = 'D:/hossein.haghighat/Features_CSV/LinearSVM';
resultFile     = fullfile(baseFeaturePath,'LOO_GreedySVM_Results.csv');

bands = 1:10;
featureNames = {'Energy','Entropy','Variance','Skewness', ...
                'Kurtosis','RMS','Hjorth_Mobility','Hjorth_Complexity'};

nBands    = numel(bands);
nFeatures = numel(featureNames);
nIC       = 21;

clip01 = @(x) max(0,min(1,x));   % safety clip

%% ================= Load Results =================
Res = readtable(resultFile);

%% ================= Output =================
ImportanceScores = zeros(nFeatures,nBands);

%% =====================================================
%  Build Optimized FIS (Full 3x3x3 rule coverage)
%% =====================================================
fis = mamfis('Name','FeatureRankingFIS_Optimized');

% -------- Inputs ----------
fis = addInput(fis,[0 1],'Name','Perf');
fis = addInput(fis,[0 1],'Name','Stab');
fis = addInput(fis,[0 1],'Name','Net');

% -------- Output ----------
fis = addOutput(fis,[0 1],'Name','Rank');

% -------- Membership Functions ----------
mfNames = {'Low','Medium','High'};

% Performance
fis = addMF(fis,'Perf','trapmf',[0 0 0.3 0.5],'Name','Low');
fis = addMF(fis,'Perf','trimf',[0.3 0.5 0.7],'Name','Medium');
fis = addMF(fis,'Perf','trapmf',[0.5 0.7 1 1],'Name','High');

% Stability
fis = addMF(fis,'Stab','trapmf',[0 0 0.3 0.5],'Name','Low');
fis = addMF(fis,'Stab','trimf',[0.3 0.5 0.7],'Name','Medium');
fis = addMF(fis,'Stab','trapmf',[0.5 0.7 1 1],'Name','High');

% Network Concentration
fis = addMF(fis,'Net','trapmf',[0 0 0.3 0.5],'Name','Low');
fis = addMF(fis,'Net','trimf',[0.3 0.5 0.7],'Name','Medium');
fis = addMF(fis,'Net','trapmf',[0.5 0.7 1 1],'Name','High');

% Output Rank
fis = addMF(fis,'Rank','trapmf',[0 0 0.3 0.5],'Name','Low');
fis = addMF(fis,'Rank','trimf',[0.3 0.5 0.7],'Name','Medium');
fis = addMF(fis,'Rank','trapmf',[0.5 0.7 1 1],'Name','High');

%% ================= Rule Base (Full 3x3x3 Coverage) =================
% High = 3, Medium = 2, Low = 1
% Priority logic: Perf > Stab > Net
ruleList = [];

for perf = 1:3
    for stab = 1:3
        for net = 1:3
            % Define output Rank based on simple logic
            % Weight Perf most, then Stab, then Net
            score = round((perf*0.5 + stab*0.3 + net*0.2));  % 1-3
            score = min(max(score,1),3);  % clip to 1-3
            ruleList = [ruleList; perf stab net score 1 1];
        end
    end
end

fis = addRule(fis,ruleList);



%% =====================================================
%  Main Loop
%% =====================================================
for b = bands
    fprintf('Processing Band %d\n',b);

    for f = 1:nFeatures
        fprintf('  Feature: %s\n',featureNames{f});

        %% ---------- 1) Performance ----------
        idx = strcmp(Res.Feature,featureNames{f}) & Res.Band==b;
        if any(idx)
            PerfScore = mean(Res.F1score(idx))/100;  % scale 0-1
        else
            PerfScore = 0.5;   % fallback neutral
        end
        PerfScore = clip01(PerfScore);

        %% ---------- 2) Stability ----------
        countFile = fullfile(baseFeaturePath,'FeatureSelectionCount', ...
            sprintf('Band_%d_%s_FeatureCount.csv',b,featureNames{f}));

        if exist(countFile,'file')
            C = readmatrix(countFile);
            if size(C,1) >= 2
                countVec = C(2,1:nIC);
            else
                countVec = ones(1,nIC);
            end
        else
            countVec = ones(1,nIC);
        end

        p = countVec / (sum(countVec)+eps);
        StabScore = 1 - entropy(p)/log(nIC);
        StabScore = clip01(StabScore);

        %% ---------- 3) Network Concentration ----------
        NetRaw   = sum(p.^2);
        NetScore = (NetRaw - 1/nIC)/(1 - 1/nIC);
        NetScore = clip01(NetScore);


        %% ---------- FIS Evaluation ----------
        ImportanceScores(f,b) = evalfis(fis,[PerfScore StabScore NetScore]);
    end
end

%% ================= Save =================
Tout = array2table(ImportanceScores, ...
    'VariableNames',compose('Band_%d',bands), ...
    'RowNames',featureNames);

writetable(Tout,fullfile(baseFeaturePath,...
    'Fuzzy_ImportanceScores_Mamdani_Optimized.csv'),...
    'WriteRowNames',true);

fprintf('\n✔ Optimized fuzzy importance scores computed SUCCESSFULLY.\n');

%% ================= Heatmap =================
figure;
imagesc(ImportanceScores);
colorbar;
xticks(1:nBands); xticklabels(compose('Band %d',bands));
yticks(1:nFeatures); yticklabels(featureNames);
xlabel('Frequency Bands');
ylabel('Features');
title('Optimized Mamdani Fuzzy Importance Scores');
set(gca,'FontSize',12);








