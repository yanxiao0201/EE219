function [X, X_holes] = train_test_split(X, test_state, random_state)
% test_state = 1~10
% returns 2 matrices
% X: (1 - train_size) fraction of random entries are zeroed out
% X_holes: where non-zero value means non-zero value in original matrix;
%          has same dimension as X
% original matrix X's value is not changed
% remember to modify weight matrix W based on X_holes
%
% Example:
% >> common
% Reading data...
% Calculating weight...
% >> [X, X_holes] = train_test_split(R, 1, 42);
% >> W(X_holes ~= 0) = 0;

rand('seed', random_state);
train_size = 0.9;

X(isnan(X)) = 0;

[m n] = size(X);
[len junk] = size(X(X ~= 0));
len_train = round(len * train_size);
len_test = len - len_train;

% finding all non-zero entries
nonzeros = zeros(len,2);
cnt = 1;
for i=1:m
    for j=find(X(i, :))
        nonzeros(cnt,:) = [i j];
        cnt = cnt + 1;
    end
end

idx_vals = randperm(len);

X_holes = zeros(m,n);

for k=(len_test*(test_state-1)+1):(len_test*test_state)
    coord = nonzeros(idx_vals(k),:);
    i = coord(1); j = coord(2);
    X_holes(i,j) = X(i,j);
    X(i,j) = 0;
end

end
