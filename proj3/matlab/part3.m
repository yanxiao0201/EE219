clear all; close all; clc;
addpath('nmfv1_4');

disp('Reading data...');
R=dlmread('out.csv');
% R(R == 0) = nan;
disp('10-fold cross validation...');

[row,column] = find(R>0);
[row_test,column_test] = train_test_split2(row,column,0.9,42,1);

len = length(row_test);
R_train = R;
for i = 1:len
    R_train(row_test(i),column_test(i)) = 0;
end
% Weight
W=(R_train==0);

%W=isnan(R);
%R(W)=0;
W=~W;

disp('Doing decomposition...');
[U,V,~,~,~]=wnmfrule(R_train,W,100);

R_predict = U*V;

list_A = zeros(len,1);
list_P = zeros(len,1);
for i = 1:len
   list_A(i) = R(row_test(i),column_test(i));
   list_P(i) = R_predict(row_test(i),column_test(i));
  
end

thres = linspace(min(list_A),max(list_A),20);
pre_list = zeros(20,1);
rec_list = zeros(20,1);
for i = 1:20
    fprintf('Calculating precision and recall when thres = %f\n',thres(i)); 
    [precision, recall] = ROC_point(list_A, list_P, thres(i));
    pre_list(i) = precision;
    rec_list(i) = recall;
end
plot(rec_list,pre_list)
xlabel('Recall')
ylabel('Precision')
title('Precision-recall curve for NMF-part3')
