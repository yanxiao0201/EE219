common;

%swap R and W
[R,W] = deal(W,R);
for k = [10,50,100]
    for lambda = [0.0,0.01,0.1,1]
        disp('Doing decomposition...');
        disp(['k = ', num2str(k), ', lambda = ', num2str(lambda)])
        [U,V,numIter,tElapsed,finalResidual] = wnmfrule(R,W,k,lambda);
    end
end

%Start evaluation with k=10
W = dlmread('out.csv');
% R(R == 0) = nan;
disp('Split data...');

[row,column] = find(W>0);
[row_test,column_test] = train_test_split2(row,column,0.9,42,1);

len = length(row_test);
R = (W~=0);
for i = 1:len
    W(row_test(i),column_test(i)) = 0;
end
% Weight
R_train = (W~=0);

disp('Doing decomposition...');
[U,V,~,~,~]=wnmfrule(R_train,W,10);

R_predict = U*V;

list_A = zeros(len,1);
list_P = zeros(len,1);

for i = 1:len
   list_A(i) = R(row_test(i),column_test(i));
   list_P(i) = R_predict(row_test(i),column_test(i));
  
end
avg_abs_error = mean(abs(list_P - list_A))

thres = linspace(0,1,20);
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
title('Precision-recall curve for NMF')


