clc; clear;

%% =======================
% Paths & Parameters
%% =======================
baseFeaturePath = 'D:/hossein.haghighat/Features_CSV';
bands = 1:10;
featureNames = {'Energy','Entropy','Variance','Skewness','Kurtosis',...
                'RMS','Hjorth_Mobility','Hjorth_Complexity'};
nFeatures = numel(featureNames);
nIC = 21;
patience = 3;        % ⬅️ Patience for greedy FS

results = table();

%% =======================
% Load Labels
%% =======================
tmp = readtable(fullfile(baseFeaturePath,'Band_1',[featureNames{1} '.csv']));
labels = tmp.Label(:);   % ASD=1 , TD=0
nSamples = length(labels);

%% =======================
% Output folders
%% =======================
countFolder   = fullfile(baseFeaturePath,'FeatureSelectionCount');
heatmapFolder = fullfile(baseFeaturePath,'Heatmaps');
if ~exist(countFolder,'dir'), mkdir(countFolder); end
if ~exist(heatmapFolder,'dir'), mkdir(heatmapFolder); end

%% =======================
% Start Parallel Pool
%% =======================
if isempty(gcp('nocreate'))
    parpool;
end

%% =======================
% Main Loop
%% =======================
for b = bands
    fprintf('\nProcessing Band %d\n',b);
    bandFolder = fullfile(baseFeaturePath,sprintf('Band_%d',b));

    for f = 1:nFeatures
        fprintf('  Feature: %s\n',featureNames{f});

        T = readtable(fullfile(bandFolder,[featureNames{f} '.csv']));
        X = T{:,1:nIC};

        predicted = zeros(nSamples,1);
        feature_count_all = zeros(nSamples,nIC);

        %% ========== LOO (PARALLEL) ==========
        parfor i = 1:nSamples

            % ---- Split ----
            X_train = X([1:i-1,i+1:end],:);
            y_train = labels([1:i-1,i+1:end]);
            X_test  = X(i,:);

            selected = [];
            remaining = 1:nIC;

            best_acc = 0;
            patience_cnt = 0;

            %% ----- Greedy FS with Patience -----
            while ~isempty(remaining) && patience_cnt < patience

                acc_list = zeros(1,length(remaining));

                for r = 1:length(remaining)
                    temp_sel = [selected remaining(r)];

                    mdl = fitcsvm(X_train(:,temp_sel),y_train,...
                        'KernelFunction','linear','Standardize',true);

                    cvp = cvpartition(y_train,'KFold',5); % stratified
                    cvmdl = crossval(mdl,'CVPartition',cvp);

                    acc_list(r) = 1 - kfoldLoss(cvmdl);
                end

                [mx,idx] = max(acc_list);

                if mx > best_acc
                    best_acc = mx;
                    selected = [selected remaining(idx)];
                    remaining(idx) = [];
                    patience_cnt = 0;        % reset
                else
                    patience_cnt = patience_cnt + 1;
                end
            end

            %% ----- Train & Test -----
            mdl = fitcsvm(X_train(:,selected),y_train,...
                'KernelFunction','linear','Standardize',true);

            predicted(i) = predict(mdl,X_test(:,selected));

            % ---- Feature Count (parfor-safe) ----
            local_count = zeros(1,nIC);
            local_count(selected) = 1;
            feature_count_all(i,:) = local_count;
        end

        feature_count = sum(feature_count_all,1);

        %% =======================
        % Metrics
        %% =======================
        TP = sum(predicted==1 & labels==1);
        TN = sum(predicted==0 & labels==0);
        FP = sum(predicted==1 & labels==0);
        FN = sum(predicted==0 & labels==1);

        acc  = (TP+TN)/(TP+TN+FP+FN)*100;
        prec = TP/(TP+FP+eps)*100;
        rec  = TP/(TP+FN+eps)*100;
        f1   = 2*(prec*rec)/(prec+rec+eps);

        %% =======================
        % Save Results
        %% =======================
        results = [results; table(b,{featureNames{f}},acc,prec,rec,f1,...
            'VariableNames',{'Band','Feature','Accuracy','Precision','Recall','F1score'})];

        countTable = array2table(feature_count,...
            'VariableNames', strcat("IC",string(1:nIC)));
        writetable(countTable, fullfile(countFolder,...
            sprintf('Band_%d_%s_FeatureCount.csv',b,featureNames{f})));
    end
end

%% =======================
% Save final results
%% =======================
writetable(results, fullfile(baseFeaturePath,'LOO_GreedySVM_Results.csv'));
disp('✔ All results saved.');

%% =======================
% Heatmaps
%% =======================
A = reshape(results.Accuracy, nFeatures, []);
F = reshape(results.F1score,  nFeatures, []);

figure;
imagesc(A); colorbar;
set(gca,'XTick',1:10,'YTick',1:nFeatures,'YTickLabel',featureNames)
xlabel('Frequency Band'); ylabel('Feature');
title('Accuracy Heatmap');
saveas(gcf, fullfile(heatmapFolder,'Accuracy_Heatmap.png'));

figure;
imagesc(F); colorbar;
set(gca,'XTick',1:10,'YTick',1:nFeatures,'YTickLabel',featureNames)
xlabel('Frequency Band'); ylabel('Feature');
title('F1-score Heatmap');
saveas(gcf, fullfile(heatmapFolder,'F1_Heatmap.png'));





