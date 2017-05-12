function [row_test,column_test] = train_test_split2(row, column, train_size, random_state, test_state)
% Example:
% R = dlmread('out.csv');
% R(R == 0) = nan;
% [X, X_test] = train_test_split(R, 0.9, 42);
% keys(X_test)

rand('seed', random_state);
num = length(row);
num_train = round(num * train_size);
num_test = num - num_train;

index_list = randperm(num);

index_min = num_test*(test_state-1)+1;
index_max = num_test*test_state;

row_test = zeros(num_test,1);
column_test = zeros(num_test,1);
%X_test = containers.Map('KeyType','int32','ValueType','any');
j = 1;
for i=index_list(index_min:index_max)
    row_test(j) = row(i);
    column_test(j) = column(i);
    j= j+1;
end

end
