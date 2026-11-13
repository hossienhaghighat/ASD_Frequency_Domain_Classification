clc; clear;

%% مسیرها
basePath = 'D:/hossein.haghighat/ASDTD';
savePath = 'D:/hossein.haghighat/Features_CSV';
if ~exist(savePath,'dir'), mkdir(savePath); end

%% شماره‌های سابجکت‌ها
subjectNumbersTD = [63:66, 75, 79:80, 87:88, 110:117, 119, 122, 124:125, ...
                    67:71, 81:86, 89:95, 118, 121, 123, ...
                    72:74, 76:77, 78, 96:109, 120];
subjectNumbersASD = 0:62;

nTD = numel(subjectNumbersTD);
nASD = numel(subjectNumbersASD);

%% بارگذاری داده‌ها
Xv = zeros(nTD,177,25);
Xvv = zeros(nASD,177,25);

for i=1:nTD
    Xv(i,:,:) = load(fullfile(basePath,sprintf('dr_stage1_subject%05d.txt',subjectNumbersTD(i))));
end

for i=1:nASD
    Xvv(i,:,:) = load(fullfile(basePath,sprintf('dr_stage1_subject%05d.txt',subjectNumbersASD(i))));
end

%% پارامترها
Fs = 0.5;
freqBands = linspace(0.001,0.1,11);
nBands = length(freqBands)-1;

bad_components = [9,15,20,25];
good_components = setdiff(1:25,bad_components);
nComp = length(good_components);

Xv = Xv(:,:,good_components);
Xvv = Xvv(:,:,good_components);

X_all = {Xv,Xvv};
nGroups = 2;

% Label vector: 1=ASD, 0=TD
labels = [zeros(nTD,1); ones(nASD,1)];

%% نام ویژگی‌ها
featureNames = {'Energy','Entropy','Variance','Skewness','Kurtosis','RMS','Hjorth_Mobility','Hjorth_Complexity'};
nFeatures = numel(featureNames);

%% شروع parallel pool
if isempty(gcp('nocreate')), parpool; end

%% استخراج ویژگی‌ها
features_all_groups = cell(nGroups,1);

for g=1:nGroups
    X = X_all{g};
    nSubjects = size(X,1);

    tmp_features = cell(nSubjects,1); % برای حل مشکل parfor

    parfor s=1:nSubjects
        data = squeeze(X(s,:,:));  % time x components
        wt_all = zeros(177,nComp,300);
        f_all = zeros(300,nComp);

        %% CWT
        for c=1:nComp
            [wt,f] = cwt(data(:,c),'amor',Fs);
            wt_all(:,c,1:length(f)) = wt.';
            f_all(1:length(f),c) = f;
        end

        % ذخیره ویژگی‌های هر سابجکت
        features_subject = zeros(nComp,nBands,nFeatures);

        for c=1:nComp
            for b=1:nBands
                f_lo = freqBands(b); f_hi = freqBands(b+1);
                f_idx = find(f_all(:,c)>=f_lo & f_all(:,c)<f_hi);
                if isempty(f_idx), continue; end

                sig = mean(abs(wt_all(:,c,f_idx)),3);
                sig = sig - mean(sig);

                % محاسبه 8 ویژگی
                energy = sum(sig.^2);
                p = sig.^2 / sum(sig.^2 + eps);
                entropy_val = -sum(p .* log(p + eps));
                variance_val = var(sig);
                skew_val = skewness(sig);
                kurt_val = kurtosis(sig);
                rms_val = sqrt(mean(sig.^2));

                dx = diff(sig);
                ddx = diff(dx);
                mobility = std(dx)/std(sig);
                complexity = (std(ddx)/std(dx))/mobility;

                feat_vals = [energy, entropy_val, variance_val, skew_val, kurt_val, rms_val, mobility, complexity];

                features_subject(c,b,:) = feat_vals;
            end
        end
        tmp_features{s} = features_subject;
    end

    features_all_groups{g} = tmp_features;
end

%% ذخیره‌سازی جداگانه بر اساس باند و ویژگی
for b=1:nBands
    bandFolder = fullfile(savePath,sprintf('Band_%d',b));
    if ~exist(bandFolder,'dir'), mkdir(bandFolder); end

    for f_idx=1:nFeatures
        featMatrix = zeros(nTD+nASD,nComp);
        for g=1:nGroups
            for s=1:numel(features_all_groups{g})
                featMatrix(s+(g-1)*numel(features_all_groups{g}),:) = features_all_groups{g}{s}(:,b,f_idx);
            end
        end
        % اضافه کردن Label
        featTable = array2table([featMatrix labels],'VariableNames',[strcat('IC',string(1:nComp)),'Label']);
        writetable(featTable, fullfile(bandFolder,sprintf('%s.csv',featureNames{f_idx})));
    end
end

fprintf('All features saved per band and per feature successfully.\n');

