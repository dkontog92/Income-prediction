clear
clc

[data, text] = xlsread('adult_50k');

%unique(text(2:end,2))
age = data(:,1);
private = strcmp(text(2:end,2),' Private');
gov = strcmp(text(2:end,2),' Local-gov')+strcmp(text(2:end,2),' State-gov')+strcmp(text(2:end,2),' Federal-gov');
self_emp = strcmp(text(2:end,2),' Self-emp-inc') + strcmp(text(2:end,2),' Self-emp-not-inc');
no_pay = strcmp(text(2:end,2),' Never-worked') + strcmp(text(2:end,2),' Without-pay');
unknown_work_class = strcmp(text(2:end,2),' ?');
married = strcmp(text(2:end,6),' Married-AF-spouse') + strcmp(text(2:end,6),' Married-civ-spouse') + strcmp(text(2:end,6),' Married-spouse-absent');
not_married = strcmp(text(2:end,6),' Divorced') + strcmp(text(2:end,6),' Never-married') + strcmp(text(2:end,6),' Separated') + strcmp(text(2:end,6),' Widowed');
white = strcmp(text(2:end,9),' White');
black = strcmp(text(2:end,9),' Black');
%other_race = strcmp(text(2:end,9),' Amer-Indian-Eskimo') + strcmp(text(2:end,9),' Asian-Pac-Islander') + strcmp(text(2:end,9),' Other');
male = strcmp(text(2:end,10),' Male');
developed_countries = strcmp(text(2:end,14),' United-States')+strcmp(text(2:end,14),' Canada')+strcmp(text(2:end,14),' England')+strcmp(text(2:end,14),' France')+strcmp(text(2:end,14),' Germany')+strcmp(text(2:end,14),' Italy')+strcmp(text(2:end,14),' Scotland')+strcmp(text(2:end,14),' Portugal');
education = data(:,5);
exec_manag = strcmp(text(2:end,7),' Exec-managerial');
prof_specialty = strcmp(text(2:end,7),' Prof-specialty');
sales = strcmp(text(2:end,7),' Sales');

y = strcmp(text(2:end,15),' >50K');

X = [age private gov self_emp no_pay unknown_work_class married ... 
    not_married white black male developed_countries education ...
    exec_manag prof_specialty sales];



X = [ones(size(X,1),1) X];
m = size(X,1);

iterations = 20000;
learning_rate = 0.007;



%% Logistic Regression

lambda = 0;
theta = ones(size(X,2),1);
hypothesis = sigmoid(X*theta);

J_hist = zeros(iterations,1);
temp = 0;

for j = 1:iterations

    J_hist(j,1) = LogComputeCost(X,y,theta,lambda);
 
    temp = theta(1) - learning_rate*(1/m)*sum((hypothesis-y).*X(:,1));
    theta = theta - learning_rate*(1/m)*(X'*(hypothesis-y)+lambda*theta);
    
    theta = [temp;theta(2:end)];
    hypothesis = sigmoid(X*theta);
      
end

figure
plot(1:iterations,J_hist);
xlabel('Iterations of Gradient Descent');
ylabel('J Cost function');

%By changing the threshold, 
threshold = 0.48;
accuracy = sum((hypothesis(:,1) >= threshold)==y)/size(hypothesis,1);
sprintf('Training performance: %.2f',accuracy*100)

precision = sum(((hypothesis(:,1) >= threshold)*3) == (2*y+1))/sum(y);
recall = sum(((hypothesis(:,1) >= threshold)*3) == (2*y+1))/sum(hypothesis(:,1) >= threshold);
f_score = (2*precision*recall)/(precision+recall);

sprintf('Precision: %.2f    Recall: %.2f    F-score: %.2f',precision*100, recall*100,f_score)
sprintf('Potentially a higher f-score can be achieved by changing the threshold')

%% Linear SVM
y = strcmp(text(2:end,15),' >50K')+0;

X = [age private gov self_emp no_pay unknown_work_class married ... 
    not_married white black male developed_countries education ...
    exec_manag prof_specialty sales];

%kernelFunction ('rbf' is gaussian, 'linear')
SVMModel = fitcsvm(X,y,'KernelFunction','linear','Standardize',false);
accuracy_SVM = sum(predict(SVMModel,X)==y)/size(y,1);
precision_SVM = sum((predict(SVMModel,X)*3) == (2*y+1))/sum(y);
recall_SVM = sum((predict(SVMModel,X)*3) == (2*y+1))/sum(predict(SVMModel,X));
f_score_SVM = (2*precision_SVM*recall_SVM)/(precision_SVM+recall_SVM);
sprintf('------------------Linear SVM-------------------------------') 
sprintf('Training performance: %.2f',accuracy_SVM*100)
sprintf('SVM Precision: %.2f    SVM Recall: %.2f   SVM F-score: %.2f',precision_SVM*100, recall_SVM*100,f_score_SVM)


%% Gaussian SVM

y = strcmp(text(2:end,15),' >50K')+0;

X = [age private gov self_emp no_pay unknown_work_class married ... 
    not_married white black male developed_countries education ...
    exec_manag prof_specialty sales];

%kernelFunction ('rbf' is gaussian, 'linear')
SVMModel_gauss = fitcsvm(X,y,'KernelFunction','rbf','Standardize',false);
accuracy_SVM_gauss = sum(predict(SVMModel_gauss,X)==y)/size(y,1);
precision_SVM_gauss = sum((predict(SVMModel_gauss,X)*3) == (2*y+1))/sum(y);
recall_SVM_gauss = sum((predict(SVMModel_gauss,X)*3) == (2*y+1))/sum(predict(SVMModel_gauss,X));
f_score_SVM_gauss = (2*precision_SVM_gauss*recall_SVM_gauss)/(precision_SVM_gauss+recall_SVM_gauss);

sprintf('------------------Gaussian SVM-------------------------------') 
sprintf('Training performance: %.2f',accuracy_SVM_gauss*100)
sprintf('SVM Precision: %.2f    SVM Recall: %.2f   SVM F-score: %.2f',precision_SVM_gauss*100, recall_SVM_gauss*100,f_score_SVM_gauss)
