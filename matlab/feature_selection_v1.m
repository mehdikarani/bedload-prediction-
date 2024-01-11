clc
clear
close all

file = 'D:/paper/qb/data/lab_data_normalized.xlsx';
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

res1=scores;
disp('--------------------------------------------------------------------')
%frmrm
disp('fsrmrmr: ')
[idx, scores] = fsrmrmr(feat, label);

fprintf(' %7s:', feat_name{idx});
disp('  ')
disp(scores(idx));
res2=scores;
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
res3=scores';
disp('--------------------------------------------------------------------')

% fsulaplacian
[idx, scores] = fsulaplacian(feat_array);

disp('Fsulaplacian: ')
fprintf(' %7s:', feat_name{idx});
disp('  ')
disp(scores(idx));
res4=scores;
disp('--------------------------------------------------------------------')
% relieff
k = 10; % Number of nearest neighbors
[idx, scores] = relieff(feat_array, label_array, k);

disp('Relieff: ')
fprintf(' %7s:', feat_name{idx});
disp('  ')
disp(scores(idx));
res5=scores;
disp('--------------------------------------------------------------------')


% RegressionBaggedEnsemble with oobPermutedPredictorImportance
ens = fitrensemble(feat, label, 'Method', 'Bag', 'NumLearningCycles', 50);
imp = oobPermutedPredictorImportance(ens);

[~, idx] = sort(imp, 'descend');
scores = imp(idx);

disp('Bag: ')
fprintf(' %7s:', feat_name{idx});
disp('  ')
disp(scores);
res6=imp(1:numel(feat_name));
disp('--------------------------------------------------------------------')

% RegressionEnsemble with predictorImportance
ens = fitrensemble(feat, label, 'Method', 'LSBoost', 'NumLearningCycles', 100);
imp = predictorImportance(ens);

[~, idx] = sort(imp, 'descend');
scores = imp(idx);

disp('LSBoost: ')
fprintf(' %8s:', feat_name{idx});
disp('  ')
disp(scores);
res7=imp(1:numel(feat_name));
disp('--------------------------------------------------------------------')
% RegressionTree with predictorImportance
tree = fitrtree(feat, label);
imp = predictorImportance(tree);

[~, idx] = sort(imp, 'descend');
scores = imp(idx);

disp('tree: ')
fprintf(' %8s:', feat_name{idx});
disp('  ')
disp(scores);
res8=imp(1:numel(feat_name));
disp('--------------------------------------------------------------------')

%write in a table
% Table headers
headers =feat_name;
% Results
results = [res1;res2;res3;res4;res5;res6;res7;res8];
% Create table

method={'Fsrftest';'fsrmrmr';'fsrnca';'Fsulaplacian';'Relieff';'Bag';'LSBoost';'tree'};
W=results(:,1);
S=results(:,2);
Q=results(:,3);
U=results(:,4);
H=results(:,5);
D50=results(:,6);
D84=results(:,7);
R=results(:,8);
T = table(method,W,S,Q,U,H,D50,D84,R);
for i=1:numel(feat_name)
    datai=results(:,i);
sumv = sum(datai);
% Normalize the data
normalizedData=datai/sumv;
    sum_score(i)=sum(normalizedData);
end

[B,I] = sort(sum_score,'descend');
fprintf(' %8s', feat_name{I})

% Specify the Excel file name and sheet name
filename = 'feature_selection.xlsx';
sheet = 'Sheet1';

% Write the table to Excel
writetable(T, filename, 'Sheet', sheet);
