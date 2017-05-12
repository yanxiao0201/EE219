common;

% parameters
k = 100;
random_state = 33;

err = [];

for test_state=1:10
    tic;
    
    Wt = W;
    fprintf('Splitting training and testing sets: Round %d\n', test_state);
    [X, X_holes] = train_test_split(R, test_state, random_state);
    disp('Factorizing. Please be very patient...');
    Wt(X_holes ~= 0) = 0;
    [U,V,numIter,tElapsed,finalResidual]=wnmfrule(X,Wt,k);
    %[U,V,numIter,tElapsed,finalResidual]=wnmfrule_orig(X,k);
    
    E = U * V;  % approximation
    pred = E(X_holes ~= 0);
    true = X(X_holes ~= 0);
    
    err(end + 1) = mean(abs(pred - true));
    
    toc;
end

disp('error = ');
disp(err');
fprintf('mean = %f\n', mean(err));
fprintf('max = %f\n', max(err));
fprintf('min = %f\n', min(err));
