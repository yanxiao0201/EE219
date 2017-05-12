common;

% parameters
k = 10;
lambda = 0.01;
random_state = 42;
L_set = [1 5:5:100];

for test_state=1:10
    tic;
    Wt = W;
    
    fprintf('Splitting training and testing sets: Round %d\n', test_state);
    [X, X_holes] = train_test_split(R, test_state, random_state);
    Wt(X_holes ~= 0) = 0;
    
    [X,Wt]=deal(Wt,X);
    disp('Factorizing. Please be very patient...');
    [U,V,numIter,tElapsed,finalResidual]=wnmfrule(X,Wt,k,lambda);
    
    E = U * V;  % approximation
    
    % create hit stats collector
    hits = containers.Map('KeyType','int32','ValueType','any');
    for L=L_set
        hits(L) = [];
    end
    
    E_size = size(E);
    for rownum=1:E_size(1)
        [sorted_row, indices] = sort(E(rownum,:), 'descend');
        test_movies = X_holes(rownum,:);
        num_movies = length(test_movies(test_movies > 0));
        test_movies(test_movies == 0) = nan;
        liked_movie_indices = find(test_movies >= 4);
        not_liked_movie_indices = find(test_movies <= 3);
    
        for L=L_set
            topL_indices = indices(1:L);
            num_hit = length(intersect(topL_indices, liked_movie_indices));
            num_false_alarm = length(intersect(topL_indices, not_liked_movie_indices));
            arr = [num_movies num_hit num_false_alarm];
            
            if isempty(hits(L))
                hits(L) = arr;
            else
                hits(L) = hits(L) + arr;
            end
        end
    end
    
    toc;
end

disp('');
disp('Report:');

hit_rate_arr = [];
false_alarm_rate_arr = [];
for L=L_set
    arr = hits(L);
    num_movies = arr(1);
    num_hit = arr(2);
    num_false_alarm = arr(3);
    
    hit_rate = num_hit / num_movies;
    false_alert_rate = num_false_alarm / num_movies;
    
    hit_rate_arr(end+1) = hit_rate;
    false_alarm_rate_arr(end+1) = false_alert_rate;
    
    fprintf('L = %f    hit rate = %f    false alarm rate = %f\n', L, hit_rate, false_alert_rate);
end

plot(false_alarm_rate_arr, hit_rate_arr);

% assist lines
hold;

maxval = hit_rate(end);
a = [0 maxval];
b = a;

plot(a,b,'--g');	% y=x
a = [0 maxval / 2];
plot(a,b,'--c');    % y=2x
a = [0 maxval / 4];
plot(a,b,'--m');    % y=4x
