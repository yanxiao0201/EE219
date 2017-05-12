function [precision, recall] = ROC_point(list_A, list_P ,thres)


pre_P = 0;
pre_A = 0;
rec_P = 0;
rec_A = 0;

for i=1:length(list_P)
    if list_P(i) > thres
        pre_P = pre_P + 1;
        if list_A(i) > thres
            pre_A = pre_A + 1;
        end 
    end
    
    if list_A(i) > thres
        rec_A = rec_A + 1;
        if list_P(i) > thres
            rec_P = rec_P + 1;
        end
    end
end

precision = pre_A/pre_P;
recall = rec_P/rec_A;


  

end