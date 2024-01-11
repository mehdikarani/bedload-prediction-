clc 
clear
close all

file = 'D:/paper/qb/data/lab_data.xlsx';

% Read the Excel file into a table
data = readtable(file);

normalizedData=data;
for i=2:numel(data(1,:))
% Calculate the mean and standard deviation
datai=table2array(data(:,i));
% Normalize the data
normalizedData(:,i) = array2table(normalize(datai));
end

file = 'D:/paper/qb/data/lab_data_normalized.xlsx';

writetable(normalizedData, file);



