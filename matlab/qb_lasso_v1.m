clc
clear
close all

file = 'D:/paper/qb/data/lab_data_pre.xlsx';
data = readtable(file);
feat_name = {'W', 's', 'Q', 'U', 'H', 'D50', 'D84','R'};
feat = data(:, 2:end-1);
label = data(:, end);

% f_test
[idx, scores] = fsrftest(feat, label);

disp('Fsrftest: ')
fprintf(' %7s:', feat_name{idx});
disp('  ')
disp(scores(idx));
disp('--------------------------------------------------------------------')
%frmrm
disp('fsrmrmr: ')
[idx, scores] = fsrmrmr(feat, label);

fprintf(' %7s:', feat_name{idx});
disp('  ')
disp(scores(idx));

disp('--------------------------------------------------------------------')
disp('fsrnca :')
feat_array = table2array(data(:, 2:end-1));
label_array = table2array(data(:, end));

% Perform feature selection using fsrnca
mdl = fsrnca(feat_array, label_array);

% Get the feature scores
scores = mdl.FeatureWeights;

% Get the indices of the selected features
[~, idx] = sort(scores, 'descend');

fprintf(' %7s:', feat_name{idx'});
disp('  ')
disp(scores(idx')');
disp('--------------------------------------------------------------------')

% fsulaplacian
[idx, scores] = fsulaplacian(feat_array);

disp('Fsulaplacian: ')
fprintf(' %7s:', feat_name{idx});
disp('  ')
disp(scores(idx));
disp('--------------------------------------------------------------------')
% relieff
k = 10; % Number of nearest neighbors
[idx, scores] = relieff(feat_array, label_array, k);

disp('Relieff: ')
fprintf(' %7s:', feat_name{idx});
disp('  ')
disp(scores(idx));
disp('--------------------------------------------------------------------')


% RegressionBaggedEnsemble with oobPermutedPredictorImportance
ens = fitrensemble(feat, label, 'Method', 'Bag', 'NumLearningCycles', 50);
imp = oobPermutedPredictorImportance(ens);

[~, idx] = sort(imp, 'descend');
scores = imp(idx);

disp('oobPermutedPredictorImportance: ')
fprintf(' %7s:', feat_name{idx});
disp('  ')
disp(scores);
disp('--------------------------------------------------------------------')

% RegressionEnsemble with predictorImportance
ens = fitrensemble(feat, label, 'Method', 'LSBoost', 'NumLearningCycles', 100);
imp = predictorImportance(ens);

[~, idx] = sort(imp, 'descend');
scores = imp(idx);

disp('predictorImportance: ')
fprintf(' %8s:', feat_name{idx});
disp('  ')
disp(scores);
disp('--------------------------------------------------------------------')
% RegressionTree with predictorImportance
tree = fitrtree(feat, label);
imp = predictorImportance(tree);

[~, idx] = sort(imp, 'descend');
scores = imp(idx);

disp('predictorImportance: ')
fprintf(' %8s:', feat_name{idx});
disp('  ')
disp(scores);
