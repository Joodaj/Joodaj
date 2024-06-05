
clc;
clear;
close all;

%NOTE MODIFICATION:   1) Search Space Division 2) SCOUT BEE modification

%% Problem Definition

CostFunction=@(x) Sphere(x);        % Cost Function

nVar=5;             % Number of Decision Variables

VarSize=[2 nVar];   % Decision Variables Matrix Size

VarMin=-10;         % Decision Variables Lower Bound
VarMax= 10;         % Decision Variables Upper Bound

%% ABC Settings

MaxIt=200;              % Maximum Number of Iterations

nPop=100;               % Population Size (Colony Size)

nOnlooker=nPop;         % Number of Onlooker Bees

L=round(0.6*nVar*nPop); % Abandonment Limit Parameter (Trial Limit)

a=1;                    % Acceleration Coefficient Upper Bound

%% Initialization

% Empty Bee Structure
empty_bee.Position=[];
empty_bee.Cost=[];

% Initialize Population Array
pop=repmat(empty_bee,nPop,1);

% Initialize Best Solution Ever Found
BestSol.Cost=inf;

% Create Initial Population
for i=1:nPop
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    pop(i).Cost=CostFunction(pop(i).Position);
    if pop(i).Cost<=BestSol.Cost
        BestSol=pop(i);
    end
end

% Abandonment Counter
C=zeros(nPop,1);

% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

%% ABC Main Loop

for it=1:MaxIt
    
    % Recruited Bees
    for i=1:nPop
        
        % Choose k randomly, not equal to i
        K=[1:i-1 i+1:nPop];
        k=K(randi([1 numel(K)]));
        
        % Define Acceleration Coeff.
        phi=a*unifrnd(-1,+1,VarSize);
        
        % New Bee Position
        newbee.Position=pop(i).Position+phi.*(pop(i).Position-pop(k).Position);
        
        % Evaluation
        newbee.Cost=CostFunction(newbee.Position);
        
        % Comparision
        if newbee.Cost<=pop(i).Cost
            pop(i)=newbee;
        else
            C(i)=C(i)+1;
        end
        
    end
    
    % Calculate Fitness Values and Selection Probabilities
    F=zeros(nPop,1);
    MeanCost = mean([pop.Cost]);
    for i=1:nPop
        F(i) = exp(-pop(i).Cost/MeanCost); % Convert Cost to Fitness
    end
    P=F/sum(F);
    
    % Onlooker Bees
    for m=1:nOnlooker
        
        % Select Source Site
        i=RouletteWheelSelection(P);
        
        % Choose k randomly, not equal to i
        K=[1:i-1 i+1:nPop];
        k=K(randi([1 numel(K)]));
        
        % Define Acceleration Coeff.
        phi=a*unifrnd(-1,+1,VarSize);
        
        % New Bee Position
        newbee.Position=pop(i).Position+phi.*(pop(i).Position-pop(k).Position);
        
        % Evaluation
        newbee.Cost=CostFunction(newbee.Position);
        
        % Comparision
        if newbee.Cost<=pop(i).Cost
            pop(i)=newbee;
        else
            C(i)=C(i)+1;
        end
        
    end
    
    % Scout Bees
    for i=1:nPop
        if C(i)>=L
            pop(i).Position=unifrnd(VarMin,VarMax,VarSize);  
            % INCLUDE NEW SCORE BEE a new equation generating the new position for scout bee 
            pop(i).Cost=CostFunction(pop(i).Position);
            C(i)=0;
        end
    end
    
    % Update Best Solution Ever Found
    for i=1:nPop
        if pop(i).Cost<=BestSol.Cost
            BestSol=pop(i);
        end
    end
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    % Display Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
end
    
%% Results

figure;
%plot(BestCost,'LineWidth',2);
semilogy(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;


function fusionfeature=EnhancedABCGSA(feat)
%NOTE MODIFICATION:   1) Search Space Division 2) SCOUT BEE modification
max_it=1; 
ElitistCheck=1; 
Rpower=1;
min_flag=1; % 1: minimization, 0: maximization
F_index=1;

%% Problem Definition
[rowf colf]=size(feat);
CostFunction=@(x) fitnessfusion(feat);        % Cost Function

nVar=colf;             % Number of Decision Variables

VarSize=[1 nVar];   % Decision Variables Matrix Size

VarMin=min(min(feat),[],2);         % Decision Variables Lower Bound
VarMax=max(max(feat),[],2);         % Decision Variables Upper Bound

%% ABC Settings

MaxIt=str2double(get(handles.iterate,'string'));%50;              % Maximum Number of Iterations

nPop=str2double(get(handles.popsize,'string'));%100;               % Population Size (Colony Size)

nOnlooker=nPop;         % Number of Onlooker Bees

L=round(0.6*nVar*nPop); % Abandonment Limit Parameter (Trial Limit)

a=str2double(get(handles.accecoeff,'string'));%1;                    % Acceleration Coefficient Upper Bound

%% Initialization

% Empty Bee Structure
empty_bee.Position=[];
empty_bee.Cost=[];

% Initialize Population Array
pop=repmat(empty_bee,nPop,1);

% Initialize Best Solution Ever Found
BestSol.Cost=inf;

% Create Initial Population
for i=1:nPop
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    pop(i).Cost=CostFunction(pop(i).Position);
    if pop(i).Cost<=BestSol.Cost
        BestSol=pop(i);
    end
end

% Abandonment Counter
C=zeros(nPop,1);

% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

%% ABC Main Loop

for it=1:MaxIt
    
    % Recruited Bees
    for i=1:nPop
        
        % Choose k randomly, not equal to i
        K=[1:i-1 i+1:nPop];
        k=K(randi([1 numel(K)]));
        
        % Define Acceleration Coeff.
        phi=a*unifrnd(-1,+1,VarSize);
        
        % New Bee Position
        newbee.Position=pop(i).Position+phi.*(pop(i).Position-pop(k).Position);
        
        % Evaluation
        newbee.Cost=CostFunction(newbee.Position);
        
        % Comparision
        if newbee.Cost<=pop(i).Cost
            pop(i)=newbee;
        else
            C(i)=C(i)+1;
        end
        
    end
    
    % Calculate Fitness Values and Selection Probabilities
    %Disruptive Selection strategy
    popcost=pop.Cost;
    if popcost(1)>0
        newpopCost=1./popcost;
    else
        newpopCost=abs(popcost);
    end
    F=zeros(nPop,1);
    [rrr ccc]=size(newpopCost);
%     see1=newpopCost
%     see2=mean([pop.Cost])
    MeanCost = newpopCost-mean([pop.Cost]);
    for i=1:nPop
        F(i) = exp(-pop(i).Cost/MeanCost); % Convert Cost to Fitness
    end
    P=F/sum(F);
    
    % Onlooker Bees
    for m=1:nOnlooker
        
        % Select Source Site
        i=RouletteWheelSelection(P);
        
        % Choose k randomly, not equal to i
        K=[1:i-1 i+1:nPop];
         seek=numel(K)
        k=K(randi([1 numel(K)]));
       
        
        % Define Acceleration Coeff.
        phi=a*unifrnd(-1,+1,VarSize);
        
        % New Bee Position
        newbee.Position=pop(i).Position+phi.*(pop(i).Position-pop(k).Position);
        [row col]=size(newbee.Position);
N=row; 
down_min=min(min(newbee.Position,[],2));
up_max=max(max(newbee.Position,[],2));
[Fbest,Lbest,BestChart,MeanChart,wb]=GSA(F_index,N,max_it,ElitistCheck,min_flag,Rpower,down_min,up_max,newbee.Position);
        
        % Evaluation
        newbee.Cost=CostFunction(newbee.Position);
%         seecost=pop(i).Cost;
%         seeFbest(i)=Fbest;
        % Comparision
        if newbee.Cost<=Fbest%pop(i).Cost
            pop(i)=newbee;
        else
            C(i)=C(i)+1;
        end
        
    end
    
    % Scout Bees
    for i=1:nPop
        if C(i)>=L
            pop(i).Position=unifrnd(VarMin,VarMax,VarSize);  
            % INCLUDE NEW SCORE BEE a new equation generating the new position for scout bee 
            pop(i).Cost=CostFunction(pop(i).Position);
            C(i)=0;
        end
    end
    
    % Update Best Solution Ever Found
    for i=1:nPop
        if pop(i).Cost<=BestSol.Cost
            BestSol=pop(i);
        end
    end
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    % Display Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
end
    
%% Results
fusionfeature=BestSol.Position;
fusionfeature=fusionfeature(1:numel(fusionfeature)/2);

% figure;
% %plot(BestCost,'LineWidth',2);
% semilogy(BestCost,'LineWidth',2);
% xlabel('Iteration');
% ylabel('Best Cost');
% grid on;


% Gravitational Search Algorithm.
function [Fbest,Lbest,BestChart,MeanChart,wb]=GSA(F_index,N,max_it,ElitistCheck,min_flag,Rpower,down_min,up_max,feat)

%V:   Velocity.
%a:   Acceleration.
%M:   Mass.  Ma=Mp=Mi=M;
%dim: Dimension of the test function.
%N:   Number of agents.
%X:   Position of agents. dim-by-N matrix.
%R:   Distance between agents in search space.
%[low-up]: Allowable range for search space.
%Rnorm:  Norm in eq.8.
%Rpower: Power of R in eq.7.
 Rnorm=2; 
%get allowable range and dimension of the test function.
[low,up,dim]=objective_functions_range(F_index,down_min,up_max);
dd=randi([5 5]);

%random initialization for agents.
seesee=feat;
X=feat(:,1:dd);%initialization(feat,dim,N,up,low)

%create the best so far chart and average fitnesses chart.
BestChart=[];MeanChart=[];

V=zeros(N,dim);

for iteration=1:max_it
%     iteration
    
    %Checking allowable range. 
    X=space_bound(X,up,low); 

    %Evaluation of agents. 
    fitness=evaluateF(X,F_index); 
    
    if min_flag==1
    [best best_X]=min(fitness); %minimization.
    else
    [best best_X]=max(fitness); %maximization.
    end        
    
    if iteration==1
       Fbest=best;Lbest=X(best_X,:);
    end
    if min_flag==1
      if best<Fbest  %minimization.
       Fbest=best;Lbest=X(best_X,:);
      end
    else 
      if best>Fbest  %maximization
       Fbest=best;Lbest=X(best_X,:);
      end
    end
      
BestChart=[BestChart Fbest];
MeanChart=[MeanChart mean(fitness)];

%Calculation of M. eq.14-20
[M]=massCalculation(fitness,min_flag); 

%Calculation of Gravitational constant. eq.13.
G=Gconstant(iteration,max_it); 

%Calculation of accelaration in gravitational field. eq.7-10,21.
a=Gfield(M,X,G,Rnorm,Rpower,ElitistCheck,iteration,max_it);

%Agent movement. eq.11-12
[X,V]=move(X,a,V);
Fbest=Fbest/2;
Lbest=Lbest/2;

if Fbest==0
    Fbest=(unifrnd(min(min(feat,[],2)),min(min(feat,[],2))+((min(min(feat,[],2)))/4),1,1))/2;
else
    Fbest=min(min(feat,[],2))/2;
end %iteration

if Lbest(1)>=0
    Lbest=(unifrnd(max(max(feat,[],2)),max(max(feat,[],2))+((max(max(feat,[],2)))/4),1,1))/3;
else
    Lbest=max(max(feat,[],2))-max(max(feat,[],2))/3;
end    
    wb=randperm(5);
end %iteration
