function [bestDesign, bestObjective, history] = SFS_Spring_Demo()
% SFS_SPRING_DEMO  Stochastic Fractal Search for constrained spring design.
%
% Source: H. Salimi, "Stochastic Fractal Search: A powerful metaheuristic
% algorithm," Knowledge-Based Systems 75 (2015), equations (11)-(18),
% (20)-(22), Tables 12-13. This code is adapted from the author's SFS
% MATLAB implementation 
%
% IMPORTANT REPRODUCIBILITY DISTINCTIONS:
%
% To run: save as SFS_Spring_Demo.m, then enter SFS_Spring_Demo in MATLAB.
% [bestDesign, bestObjective, history] = SFS_Spring_Demo;
% No Statistics and Machine Learning Toolbox required (uses randn).

clc;

%% SETTINGS: these may be changed before class
rng(42, 'twister');           % Reproducible random sequence
N = 100;                     % Number of particles; 2015 Table 4 setting
maxGenerations = 800;        % Stop after this many generations
MDN = 1;                     % Maximum Diffusion Number; Table 4 setting
probWalk1 = 0.5;             % Prob. of choosing Eq. 11 (else Eq. 12)
USE_POSTED_RANDOMNESS = false; % false: printed paper; true: posted code
showAnimation = true;
plotEvery = 10;              % Draw every 10 generations
printEvery = 50;             % Command Window reporting interval
feasibilityTolerance = 1e-12;

% Design vector P = [w, d, L]
LB = [0.05, 0.25, 2.0];
UB = [2.00, 1.30, 15.0];
D = numel(LB);

% Table 12: published SFS results [w,d,l] and objective function value
paperDesign = [0.051689060916152, ...
               0.356717735791209, ...
              11.288965986603480];
paperObjective = 0.012665232788319;

%% STEP 1: INITIALIZE THE PARTICLES -- Salimi Eq. (14)
P = repmat(LB, N, 1) + rand(N,D).*repmat(UB-LB, N, 1);
[f, violation, ~] = evaluateSpring(P);
[P, f, violation] = sortPopulation(P, f, violation, feasibilityTolerance);

bestDesign = P(1,:);
bestObjective = f(1);
bestViolation = violation(1);
history = NaN(maxGenerations+1,1);
if bestViolation <= feasibilityTolerance
    history(1) = bestObjective;
end

%% SET UP LIVE
if showAnimation
    fig = figure('Name','SFS: Tension / Compression Spring', ...
        'Color','b','NumberTitle','off');
    axPop = subplot(1,2,1);
    hold(axPop,'on');
    hInfeasible = scatter3(axPop,NaN,NaN,NaN,17,[0.70 0.70 0.70],'filled');
    hFeasible = scatter3(axPop,NaN,NaN,NaN,25,[0.04 0.48 0.72],'filled');
    hBest = scatter3(axPop,NaN,NaN,NaN,105,[0.88 0.22 0.14],'p','filled');
    xlabel(axPop,'Wire diameter w');
    ylabel(axPop,'Coil diameter d');
    zlabel(axPop,'Coil parameter L');
    title(axPop,'Candidate spring designs');
    axis(axPop,[LB(1) UB(1) LB(2) UB(2) LB(3) UB(3)]);
    grid(axPop,'on');
    view(axPop,35,22);
    legend(axPop,{'Infeasible','Feasible','Best feasible'},'Location','best');

    axCurve = subplot(1,2,2);
    hold(axCurve,'on');
    hHistory = semilogy(axCurve,NaN,NaN,'b-','LineWidth',2);
    semilogy(axCurve,[0 maxGenerations],[paperObjective paperObjective], ...
        'r--','LineWidth',1.4);
    xlabel(axCurve,'Generation');
    ylabel(axCurve,'Best feasible objective f(w,d,L)');
    title(axCurve,'SFS convergence');
    legend(axCurve,{'SFS result','Salimi (2015) Table 12'},'Location','northeast');
    grid(axCurve,'on');
    xlim(axCurve,[0 maxGenerations]);
    drawnow;
end

fprintf('\nSFS CONSTRAINED SPRING DESIGN\n');
fprintf('Population = %d | Generations = %d | MDN = %d\n',N,maxGenerations,MDN);
fprintf('Constraint comparison: feasibility first (adaptation).\n');
if USE_POSTED_RANDOMNESS
    fprintf('Random terms: as used in the author-posted code.\n');
else
    fprintf('Random terms: as in the 2015 printed equations.\n');
end
fprintf('---------------------------------------------------------------\n');

tStart = tic;

%% STEP 2: DIFFUSION AND TWO UPDATING PROCESSES
for g = 1:maxGenerations
    % The printed sigma formula is zero at g=1. Using g+1 avoids a
    % zero-spread first diffusion; this is a documented implementation
    scale = log(g+1)/(g+1);

    % DIFFUSION -- Salimi Eqs. (11)-(13)
    Q = zeros(N,D);
    currentBest = P(1,:);
    for i = 1:N
        sigma = abs(scale.*(P(i,:)-currentBest));   % Eq. (13)
        trials = zeros(MDN+1,D);
        trials(1,:) = P(i,:);                      % Include parent point
        for k = 1:MDN
            if rand < probWalk1
                % Eq. (11): best-point-referenced Gaussian walk
                if USE_POSTED_RANDOMNESS
                    % Author's uploaded code uses Gaussian coefficients.
                    shift = randn*currentBest-randn*P(i,:);
                    mu = currentBest;
                else
                    % Printed paper uses UNIFORM epsilon & epsilon-prime.
                    shift = rand*currentBest-rand*P(i,:);
                    mu = abs(currentBest);
                end
                trials(k+1,:) = mu + sigma.*randn(1,D) + shift;
            else
                % Eq. (12): particle-referenced Gaussian walk
                if USE_POSTED_RANDOMNESS
                    mu = P(i,:);
                else
                    mu = abs(P(i,:));
                end
                trials(k+1,:) = mu + sigma.*randn(1,D);
            end
        end
        trials = repairBounds(trials,LB,UB);
        [ft,vt,~] = evaluateSpring(trials);
        order = candidateOrder(ft,vt,feasibilityTolerance);
        Q(i,:) = trials(order(1),:);               % Best trial survives
    end
    [f,violation,~] = evaluateSpring(Q);
    [P,f,violation] = sortPopulation(Q,f,violation,feasibilityTolerance);

    % FIRST UPDATING -- Salimi Eqs. (15)-(16)
    % Sorted best-to-worst: rank(best)=N and rank(worst)=1.
    Pa = (N:-1:1)'/N;
    r1 = randperm(N);
    r2 = randperm(N);
    proposal = P;
    for i = 1:N
        for j = 1:D
            if rand > Pa(i)
                proposal(i,j) = P(r1(i),j) - rand*(P(r2(i),j)-P(i,j));
            end
        end
    end
    proposal = repairBounds(proposal,LB,UB);
    [newF,newV,~] = evaluateSpring(proposal);
    for i = 1:N
        if isBetter(newF(i),newV(i),f(i),violation(i),feasibilityTolerance)
            P(i,:) = proposal(i,:);
            f(i) = newF(i);
            violation(i) = newV(i);
        end
    end
    [P,f,violation] = sortPopulation(P,f,violation,feasibilityTolerance);

    % SECOND UPDATING -- Salimi Eqs. (15), (17), (18)
    Pa = (N:-1:1)'/N;
    currentBest = P(1,:);
    for i = 1:N
        if rand > Pa(i)
            r = randi(N);
            t = randi(N);
            while r == t
                t = randi(N);
            end
            if USE_POSTED_RANDOMNESS
                epsHat = rand;          % Author's posted MATLAB code
            else
                epsHat = randn;         % Printed Eqs. (17) and (18)
            end
            if rand <= 0.5
                trial = P(i,:) - epsHat*(P(t,:)-currentBest); % Eq. 17
            else
                trial = P(i,:) + epsHat*(P(t,:)-P(r,:));       % Eq. 18
            end
            trial = repairBounds(trial,LB,UB);
            [trialF,trialV,~] = evaluateSpring(trial);
            if isBetter(trialF,trialV,f(i),violation(i),feasibilityTolerance)
                P(i,:) = trial;
                f(i) = trialF;
                violation(i) = trialV;
            end
        end
    end
    [P,f,violation] = sortPopulation(P,f,violation,feasibilityTolerance);

    % Track the global best, across both updating stages.
    if isBetter(f(1),violation(1),bestObjective,bestViolation,feasibilityTolerance)
        bestDesign = P(1,:);
        bestObjective = f(1);
        bestViolation = violation(1);
    end
    if bestViolation <= feasibilityTolerance
        history(g+1) = bestObjective;
    end

    if mod(g,printEvery)==0 || g==1 || g==maxGenerations
        fprintf('Generation %4d | f = %.12f | total violation = %.3g\n', ...
            g,bestObjective,bestViolation);
    end

    if showAnimation && isgraphics(fig) && (mod(g,plotEvery)==0 || g==1 || g==maxGenerations)
        feasible = (violation<=feasibilityTolerance);
        set(hFeasible,'XData',P(feasible,1),'YData',P(feasible,2), ...
                      'ZData',P(feasible,3));
        set(hInfeasible,'XData',P(~feasible,1),'YData',P(~feasible,2), ...
                        'ZData',P(~feasible,3));
        if bestViolation<=feasibilityTolerance
            set(hBest,'XData',bestDesign(1),'YData',bestDesign(2), ...
                      'ZData',bestDesign(3));
        end
        set(hHistory,'XData',0:g,'YData',history(1:g+1));
        title(axCurve,sprintf('Generation %d | Best f = %.8f',g,bestObjective));
        drawnow limitrate;
    end
end
elapsed = toc(tStart);

%% FINAL VALIDATION AGAINST THE PUBLISHED SPRING RESULT
[~,finalViolation,constraints] = evaluateSpring(bestDesign);
fprintf('\n================ FINAL SFS RESULTS ================\n');
fprintf('w (wire diameter)         = %.12f\n',bestDesign(1));
fprintf('d (mean coil diameter)    = %.12f\n',bestDesign(2));
fprintf('L (length / coil param.)  = %.12f\n',bestDesign(3));
fprintf('Objective f(w,d,L)        = %.12f\n',bestObjective);
fprintf('Total constraint violation= %.4e\n',finalViolation);
if finalViolation > feasibilityTolerance
    warning('No fully feasible spring found; this result is NOT a valid optimum.');
end
for k=1:4
    fprintf('Constraint g%d = %+0.8e (must be <= 0)\n',k,constraints(k));
end
fprintf('Paper f (Table 12)        = %.12f\n',paperObjective);
fprintf('Absolute objective gap    = %.8e\n',abs(bestObjective-paperObjective));
fprintf('Elapsed computation       = %.3f seconds\n',elapsed);
fprintf('===================================================\n');
end

% ---------------------------------------------------------------
function [f,violation,G] = evaluateSpring(P)
% Salimi (2015), Eqs. (20)-(22). Each row P(i,:) = [w_i,d_i,L_i].
w = P(:,1);
d = P(:,2);
L = P(:,3);

f = (L+2).*w.^2.*d;
g1 = 1 - d.^3.*L./(71785*w.^4);
g2 = d.*(4*d-w)./(12566*w.^3.*(d-w)) + 1./(5108*w.^2) - 1;
g3 = 1 - 140.45*w./(d.^2.*L);
g4 = 2*(w+d)/3 - 1;
G = [g1 g2 g3 g4];
violation = sum(max(G,0),2);  % Zero if all four constraints satisfied
bad = ~isfinite(f) | any(~isfinite(G),2);
f(bad) = Inf;
violation(bad) = Inf;
end

% ---------------------------------------------------------------
function P = repairBounds(P,LB,UB)
% Author's Bound_Checking strategy: randomly reset each invalid component.
for j=1:numel(LB)
    bad = (P(:,j)<LB(j)) | (P(:,j)>UB(j)) | ~isfinite(P(:,j));
    P(bad,j) = LB(j) + rand(sum(bad),1)*(UB(j)-LB(j));
end
end

% ---------------------------------------------------------------
function idx = candidateOrder(f,v,tol)
% OUR constraint handling 
% feasible beats infeasible; among feasible lower f is better;
% among infeasible smaller sum of positive constraint violations wins.
[~,idx] = sortrows([double(v>tol),v,f],[1 2 3]);
end

% ---------------------------------------------------------------
function [P,f,v] = sortPopulation(P,f,v,tol)
idx = candidateOrder(f,v,tol);
P = P(idx,:);
f = f(idx);
v = v(idx);
end

% ---------------------------------------------------------------
function yes = isBetter(newF,newV,oldF,oldV,tol)
newFeasible = newV<=tol;
oldFeasible = oldV<=tol;
if newFeasible && ~oldFeasible
    yes = true;
elseif ~newFeasible && oldFeasible
    yes = false;
elseif newFeasible && oldFeasible
    yes = newF<oldF;
else
    yes = (newV<oldV) || (newV==oldV && newF<oldF);
end
end
