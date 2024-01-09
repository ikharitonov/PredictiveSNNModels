%--------------------------------------------------------------------------
%                       recurrent neural network 
%--------------------------------------------------------------------------
% t=1 ah(:,1) = ah0       + (dt./Tau).*(-ah0       + Whx*IN(:,1) + Whh*h0       + bahneverlearn(:,t) + bah)       hidden activation
% t>1 ah(:,t) = ah(:,t-1) + (dt./Tau).*(-ah(:,t-1) + Whx*IN(:,t) + Whh*h(:,t-1) + bahneverlearn(:,t) + bah)       hidden activation
%      h(:,t) = f(ah(:,t)) + bhneverlearn(:,t)                             hidden units
%     ay(:,t) = Wyh*h(:,t) + bayneverlearn(:,t) + bay                      output activation
%      y(:,t) = g(ay(:,t))                                                 output units
%--------------------------------------------------------------------------
% IN  - dimIN x numT x numexamples matrix, inputs, IN(:,i,j) is an input vector at time T(i) trial j 
% y   - dimOUT x numT x numexamples matrix, outputs, y(:,i,j) is the output vector at time T(i) trial j
% f   - hidden unit nonlinearity 
% g   - output unit nonlinearity 

% Whx - numh x dimIN matris, input-to-hidden weight matrix
% Whh - numh x numh matrix, hidden-to-hidden weight matrix
% Wyh - dimOUT x numh matrix, hidden-to-output weight matrix
% bah - numh x 1 matrix, hidden bias
% bay - dimOUT x 1 matrix, output bias
% Tau - numh x 1 matrix

% ah0 - numh x numexamples matrix, initial activation of hidden units
% h0  - numh x numexamples matrix, initial values of hidden units
% bahneverlearn - numh x numT x numexamples matrix, hidden activation bias, set to 0 instead of zeros(numh,numT,numexamples) to save memory and run faster, code is written so this only works if all never-learned biases are 0
% bhneverlearn  - numh x numT x numexamples matrix, hidden bias, set to 0 instead of zeros(numh,numT,numexamples) to save memory and run faster, code is written so this only works if all never-learned biases are 0
% bayneverlearn - dimOUT x numT x numexamples matrix, output activation bias, set to 0 instead of zeros(dimOUT,numT,numexamples) to save memory and run faster, code is written so this only works if all never-learned biases are 0
close all; 
clear all;
%---------------
figuredir = '/Users/christophercueva/Desktop/neural network train/';% save figures to figuredir
%---------------
datadir = figuredir;
addpath(genpath(figuredir))% add folder and all of its subfolders to the top of the search path
cd(datadir)
SAVE = 1;% if SAVE = 1 save parameters
PLOTFIGURESDURINGTRAINING = 1;% if 1 plot figures during training
DISPLAYON = 0;

dimIN = 3;% number of inputs
numh = 100;% number of hidden units
dimOUT = 2;% number of outputs
numparameters = 0;
LEARNPARAMETERS_Whx = 1; if LEARNPARAMETERS_Whx==1; numparameters = numparameters + numh*dimIN; end% if 1 learn parameter, if 0 treat as constant
LEARNPARAMETERS_Whh = 1; if LEARNPARAMETERS_Whh==1; numparameters = numparameters + numh*numh; end% if 1 learn parameter, if 0 treat as constant
LEARNPARAMETERS_Wyh = 1; if LEARNPARAMETERS_Wyh==1; numparameters = numparameters + dimOUT*numh; end% if 1 learn parameter, if 0 treat as constant
LEARNPARAMETERS_bah = 1; if LEARNPARAMETERS_bah==1; numparameters = numparameters + numh; end% if 1 learn parameter, if 0 treat as constant
LEARNPARAMETERS_bay = 0; if LEARNPARAMETERS_bay==1; numparameters = numparameters + dimOUT; end% if 1 learn parameter, if 0 treat as constant
LEARNPARAMETERS_Tau = 0; if LEARNPARAMETERS_Tau==1; numparameters = numparameters + numh; end% if 1 learn parameter, if 0 treat as constant

L1REGULARIZE_Whx = 0;% if 1 regularize with lambdaL1, larger lambdaL1 = more regularization = smaller parameters
L1REGULARIZE_Whh = 0;% if 1 regularize with lambdaL1, larger lambdaL1 = more regularization = smaller parameters
L1REGULARIZE_Wyh = 0;% if 1 regularize with lambdaL1, larger lambdaL1 = more regularization = smaller parameters
L1REGULARIZE_bah = 0;% if 1 regularize with lambdaL1, larger lambdaL1 = more regularization = smaller parameters
L1REGULARIZE_bay = 0;% if 1 regularize with lambdaL1, larger lambdaL1 = more regularization = smaller parameters
L1REGULARIZE_Tau = 0;% if 1 regularize with lambdaL1, larger lambdaL1 = more regularization = smaller parameters
% this code is setup so some parameters must have L2 regularization, otherwise EL2 is 0 causing lambdaL2 to grow and hence lambdaSD to grow and then the parameters do not change
L2REGULARIZE_Whx = 1;% if 1 regularize with lambdaL2, larger lambdaL2 = more regularization = smaller parameters
L2REGULARIZE_Whh = 0;% if 1 regularize with lambdaL2, larger lambdaL2 = more regularization = smaller parameters
L2REGULARIZE_Wyh = 1;% if 1 regularize with lambdaL2, larger lambdaL2 = more regularization = smaller parameters
L2REGULARIZE_bah = 0;% if 1 regularize with lambdaL2, larger lambdaL2 = more regularization = smaller parameters
L2REGULARIZE_bay = 0;% if 1 regularize with lambdaL2, larger lambdaL2 = more regularization = smaller parameters
L2REGULARIZE_Tau = 0;% if 1 regularize with lambdaL2, larger lambdaL2 = more regularization = smaller parameters

numtrain = 500;% 10000, number of training trials 
numtrainGv = 500;% 1000, number of training trials used to calculate G*v, must be less than or equal to numtrain
numtrialstest = 0;% 1000, errormain-test is calculated, and then displayed, on numtrialstest trials during training epochs
dt = 1; Tau = 10*ones(numh,1);% neural time-constant 
numT = 500;% number of time-steps in simulation
numTtest = numT;% number of time-steps in test set
nonlinearity{1} = 'retanh';% nonlinearity{1} is the nonlinearity of the hidden units - options: 'linear' 'logistic' 'tanh' 'retanh' 'ReLU' 'ELU'
nonlinearity{2} = 'linear';% nonlinearity{2} is the nonlinearity of the output units - options: 'linear' 'logistic' 'tanh' 'retanh' 'ReLU' 'ELU'
 
NOISEAMPLITUDE = 0.1;% standard deviation of firing rate noise, bhneverlearn = NOISEAMPLITUDE*randn(numh,numT,numtrain);
% parameters for generateINandTARGETOUT
ANGULARSPEED = 1;
angle0duration = 10;% angle0 input (sin(angle0) and cos(angle0)) is nonzero for angle0duration timesteps at beginning of trial
%ANGULARVELOCITY.discreteangularvelocitydegrees = [-8 0 0 0 8];% if this variable exists then the angular velocity is drawn randomly from this discrete set at each timestep, and angularmomentum and sd are not used
ANGULARVELOCITY.angularmomentum = 0.8;% angularvelocity(t) = sd*randn + angularmomentum * angularvelocity(t-1), angularmomentum and sd are not used when using ANGULARVELOCITY.discreteangularvelocitydegrees
ANGULARVELOCITY.sd = ANGULARSPEED * 0.03;% at each timestep the new angularvelocity is a gaussian random variable with mean 0 and standard deviation sd, angularmomentum and sd are not used when using ANGULARVELOCITY.discreteangularvelocitydegrees
ANGULARVELOCITY.angularvelocitymindegrees = -Inf;% minimum angular input/angular-velocity on a single timestep (degrees)
ANGULARVELOCITY.angularvelocitymaxdegrees = Inf;% maximum angular input/angular-velocity on a single timestep (degrees)
noiseamplitude_input = 0;
BOUNDARY.periodic = 1;% if 1 periodic boundary conditions
% if BOUNDARY.periodic = 0 specify BOUNDARY.minangle and BOUNDARY.maxangle 
PULSEINPUT.numtrials = 0;


%--------------------------------------------------------------------------
%          distribution of angular input used during training
%--------------------------------------------------------------------------
%numtrain = 50000;
randseed = 1;
[IN, TARGETOUT, itimeRNN] = generateINandTARGETOUT(dimIN,dimOUT,numT,numtrain,randseed,noiseamplitude_input,angle0duration,ANGULARVELOCITY,BOUNDARY);
%A = IN(1,:,:)*180/pi; stdA = std(A(:))% 2.8293 when speed=1, mom=0.8
    
eps = 10^-4;% add epsilon to largest element of anglegrid so when angle equals max(anglegrid) then iangle is not empty 
minangularinput = min(min(IN(1,:,:))) * 180/pi; maxangularinput = max(max(IN(1,:,:))) * 180/pi; dangle = (maxangularinput - minangularinput)/100; anglegrid = [minangularinput:dangle:(maxangularinput-dangle) maxangularinput+eps];% 1 x something matrix, discretize the angular input
numvisits = zeros(1,numel(anglegrid)-1);% number of times agent visits bin
for itrial=1:numtrain
    for t=1:numT
        angle = IN(1,t,itrial)*180/pi;% angular input in degrees
        iangle = find(angle < anglegrid,1,'first') - 1;% if angle = min(anglegrid) iangle = 1, if angle = maxangularinput iangle is numel(anglegrid)-1
        numvisits(iangle) = numvisits(iangle) + 1;
    end
end
if sum(numvisits(:)) ~= numtrain*numT; error('missing visits'); end
    
handle = figure;% input distribution
clf; hold on; fontsize = 33; set(gcf,'DefaultLineLineWidth',6,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
plot(anglegrid(1:end-1)+dangle/2,numvisits,'k-')    
xlabel((sprintf('Angular input (%.3g to %.3g degrees)',min(anglegrid),max(anglegrid))))
ylabel('Frequency')   
title({['Distribution of angular inputs to RNN'];[sprintf('%g training trials, %g timesteps',numtrain,numT)]})
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
axis tight; axis([-Inf Inf 0 max(ylim)+abs(max(ylim)-min(ylim))/100])
set(gca,'linewidth',2); set(gca,'TickLength',[0.02 0.025])% default set(gca,'TickLength',[0.01 0.025])
print(handle, '-dpdf', sprintf('%s/inputstatistics_%gtrainingtrials_numT%g',figuredir,numtrain,numT))

handle = figure;% input distribution, all figures (speed 0.1,1,3) should have same x-axis limits
clf; hold on; fontsize = 33; set(gcf,'DefaultLineLineWidth',6,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
plot(anglegrid(1:end-1)+dangle/2,numvisits,'k-')    
xlabel((sprintf('Angular input (%.3g to %.3g degrees)',min(anglegrid),max(anglegrid))))
ylabel('Frequency')   
title({['Distribution of angular inputs to RNN'];[sprintf('%g training trials, %g timesteps',numtrain,numT)]})
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
%axis tight; axis([-Inf Inf 0 max(ylim)+abs(max(ylim)-min(ylim))/100])
axis tight; axis([min(xlim) max(xlim) 0 max(ylim)+abs(max(ylim)-min(ylim))/100])
xlim([-39 39]); handleaxis = gca; handleaxis.YAxis.Visible = 'off';
set(gca,'linewidth',2); set(gca,'TickLength',[0.02 0.025])% default set(gca,'TickLength',[0.01 0.025])
print(handle, '-dpdf', sprintf('%s/inputstatistics_%gtrainingtrials_numT%g_',figuredir,numtrain,numT))

%--------------------------------------------------------------------------
%        autocorrelation of angular velocity input used during training
%--------------------------------------------------------------------------
randseed = 1;
numT_ = 5000;
numtrain_ = 202;% analyze trial without long period of zero input
[IN, TARGETOUT, itimeRNN] = generateINandTARGETOUT(dimIN,dimOUT,numT_,numtrain_,randseed,noiseamplitude_input,angle0duration,ANGULARVELOCITY,BOUNDARY);
% there are 3 inputs, 
% 1) angular velocity, angle to integrate
% 2) sin(angle0)
% 3) cos(angle0)

x = IN(1,angle0duration + 1:end,201); figure; plot(x*180/pi);% analyze trial without long period of zero input
% mean(abs(x*180/pi)) is 2.2308 degrees/timestep * 1timestep/20ms * 1000ms/1sec = 111.5394 degrees/sec
% mean(abs(x*180/pi)) is 2.2308 degrees/timestep * 1timestep/25ms * 1000ms/1sec = 89.232 degrees/sec

numx = length(x);
Rxx = zeros(1,numx);% autocorrelation of x at lag [0:numx-1]
for lag=0:numx-1
    Rxx(lag+1) = x(1:numx-lag) * x(lag+1:numx)';
end
Rxxmatlab = xcorr(x);
Rxxmatlab = Rxxmatlab(end-numx+1:end);

figure; hold on;
plot([0:numx-1],Rxx,'k-')
plot([0:numx-1],Rxxmatlab,'r--')
xlabel('Lag')
ylabel('Autocorrelation')

handle = figure;
clf; hold on; fontsize = 33; set(gcf,'DefaultLineLineWidth',6,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
plot([0:20],Rxx(1:21)/Rxx(1),'k-')
xlabel('Lag')
ylabel('Autocorrelation')
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
axis tight; axis([-Inf Inf 0 max(ylim)+abs(max(ylim)-min(ylim))/100])
set(gca,'linewidth',2); set(gca,'TickLength',[0.02 0.025])% default set(gca,'TickLength',[0.01 0.025])
print(handle, '-dpdf', sprintf('%s/autocorrelationofangularvelocityinput_numT%g',figuredir,numT_))
%--------------------------------------------------------------------------



rng(11); Whx = randn(numh,dimIN) / sqrt(dimIN);% numh x dimIN matrix

% Saxe at al. 2014 "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"
% We empirically show that if we choose the initial weights in each layer to be a random orthogonal matrix (satisifying W'*W = I), instead of a scaled random Gaussian matrix, then this orthogonal random initialization condition yields depth independent learning times just like greedy layerwise pre-training. 
rng(1); Whh = randn(numh,numh)/sqrt(numh); [u,~,v] = svd(Whh); Whh = u*diag(1.0*ones(numh,1))*v';% make the eigenvalues large so they decay slowly
Whh(eye(numh,numh)==1) = 0;% no self connections

density = 0.15;
Wyh = sqrt(1/round(numh*density))*full(sprandn(dimOUT,numh,density));% dimOUT x numh matrix, Gaussian with mean 0 and variance 1/15
Wyh = zeros(dimOUT,numh);% dimOUT x numh matrix%bh = sqrt(1/15)*randn(numh,1) - 3;% numh x 1, constant biases for hidden units, Gaussian with mean 0 and variance 1/15
Wyh = 0.01 * randn(dimOUT,numh) / sqrt(numh);
bah = 0.1 + 0.01*randn(numh,1);% initialize with small positive values so there are fewer "dead" units
bay = zeros(dimOUT,1);


% constants
ah0 = zeros(numh,numtrain);
h0 = computenonlinearity(ah0,nonlinearity{1});
bahneverlearn = zeros(numh,numT,numtrain);% numh x numT x numexamples matrix, biases for hidden units, set to 0 instead of zeros(numh,numT,numexamples) to save memory and run faster, code is written so this only works if all never-learned biases are 0
bhneverlearn = zeros(numh,numT,numtrain);% numh x numT x numexamples matrix, biases for hidden units, set to 0 instead of zeros(numh,numT,numexamples) to save memory and run faster, code is written so this only works if all never-learned biases are 0
bayneverlearn = zeros(dimOUT,numT,numtrain);% dimOUT x numT x numexamples matrix, biases for output units, set to 0 instead of zeros(dimOUT,numT,numexamples) to save memory and run faster, code is written so this only works if all never-learned biases are 0
%bahneverlearn = 0; bhneverlearn = 0; bayneverlearn = 0;% set to 0 instead of zeros(...,numT,numexamples) to save memory and run faster, code is written so this only works if all never-learned biases are 0
ah0test = zeros(numh,numtrialstest);
h0test = computenonlinearity(ah0test,nonlinearity{1});
bahneverlearntest = zeros(numh,numTtest,numtrialstest);% numh x numT x numexamples matrix, biases for hidden units, set to 0 instead of zeros(numh,numT,numexamples) to save memory and run faster, code is written so this only works if all never-learned biases are 0
bhneverlearntest = zeros(numh,numTtest,numtrialstest);% numh x numT x numexamples matrix, biases for hidden units, set to 0 instead of zeros(numh,numT,numexamples) to save memory and run faster, code is written so this only works if all never-learned biases are 0
bayneverlearntest = zeros(dimOUT,numTtest,numtrialstest);% dimOUT x numT x numexamples matrix, biases for output units, set to 0 instead of zeros(dimOUT,numT,numexamples) to save memory and run faster, code is written so this only works if all never-learned biases are 0
%bahneverlearntest = 0; bhneverlearntest = 0; bayneverlearntest = 0;% set to 0 instead of zeros(...,numT,numexamples) to save memory and run faster, code is written so this only works if all never-learned biases are 0

handle = figure;% eigenvalues of Whh
hold on; fontsize = 15; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',10,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)% only need to set DefaulttextFontSize if using function text
scatter(real(eig(Whh)),imag(eig(Whh)),50,'markerfacecolor','k','markeredgecolor','k');
xlabel('real(eig(Whh))')
ylabel('imag(eig(Whh))')
axis tight; axis([-Inf Inf min(ylim)-abs(max(ylim)-min(ylim))/100  max(ylim)+abs(max(ylim)-min(ylim))/100])
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
print(handle, '-dpdf', sprintf('%s/eigenvaluesWhh_beforelearning',figuredir))


parameters = [];% order of parameters must be Whx, Whh, Wyh, bah, bay, Tau
if LEARNPARAMETERS_Whx==1; parameters = [parameters; Whx(:)]; end
if LEARNPARAMETERS_Whh==1; parameters = [parameters; Whh(:)]; end
if LEARNPARAMETERS_Wyh==1; parameters = [parameters; Wyh(:)]; end
if LEARNPARAMETERS_bah==1; parameters = [parameters; bah(:)]; end
if LEARNPARAMETERS_bay==1; parameters = [parameters; bay(:)]; end
if LEARNPARAMETERS_Tau==1; parameters = [parameters; Tau(:)]; end
if length(parameters)~=numparameters; error('error in parameters'); end
epoch = 0;% save initial parameters
eval(sprintf('parameters_saveepoch%g = [ah0(:); h0(:); Whx(:); Whh(:); Wyh(:); bah; bay; Tau];',epoch)); save(sprintf('parameters_saveepoch%g.mat',epoch), sprintf('parameters_saveepoch%g',epoch));


lambdaL1 = 0;% 0.5/5 L1 regularization on parameters
lambdaL2 = 0.5/2; mu = 1500*2;% L2 regularization on parameters
lambdaSD = lambdaL2*mu;
lambdahL1 = 0;% L1 regularization on h - "firing rate" of units, 0.1/2 
lambdahL2 = 0.1/2;% L2 regularization on h - "firing rate" of units, 0.1/2 

numepoch = 500;% 3000 
%epochset_saveparameters = [1:5 10:5:min(500,numepoch) 600:100:numepoch];% save parameters when epoch is a member of epochset_saveparameters
epochset_saveparameters = [1:20 25:5:min(500,numepoch) 600:100:numepoch];% save parameters when epoch is a member of epochset_saveparameters
tictoctime_store = -700*ones(1,numepoch);% tictoctime_store(j) training time (seconds) for epoch j
parametersCG = zeros(numparameters,1);
rho_store = -700*ones(1,numepoch);
error_store = -700*ones(1,numepoch); errormain_store = -700*ones(1,numepoch); errormaintest_store = -700*ones(1,numepoch); errorL1_store = -700*ones(1,numepoch); errorL2_store = -700*ones(1,numepoch); errorhL1_store = -700*ones(1,numepoch); errorhL2_store = -700*ones(1,numepoch);
normgradE_store = -700*ones(1,numepoch); normdeltaparameters_store = -700*ones(1,numepoch);
if LEARNPARAMETERS_Whh% information bottleneck
    meanabsgradEWhh_store = -700*ones(1,numepoch);
    stdgradEWhh_store = -700*ones(1,numepoch);
end
medianabsah_store = -700*ones(1,numepoch);
meanabsah_store = -700*ones(1,numepoch);
maxabsah_store = -700*ones(1,numepoch);
maxabsparameters_store = -700*ones(1,numepoch);  
meandeltah = [];% something x numT matrix
for epoch=1:numepoch% to train a RNN with a different architecture change model, unpack, forwardpass, and parameters_saveepoch
    bhneverlearn = NOISEAMPLITUDE*randn(numh,numT,numtrain);
    bhneverlearntest = NOISEAMPLITUDE*randn(numh,numTtest,numtrialstest);
    
    Whh(eye(numh,numh)==1) = 0;% no self connections
    parameters = [];% order of parameters must be Whx, Whh, Wyh, bah, bay, Tau
    if LEARNPARAMETERS_Whx==1; parameters = [parameters; Whx(:)]; end
    if LEARNPARAMETERS_Whh==1; parameters = [parameters; Whh(:)]; end
    if LEARNPARAMETERS_Wyh==1; parameters = [parameters; Wyh(:)]; end
    if LEARNPARAMETERS_bah==1; parameters = [parameters; bah(:)]; end
    if LEARNPARAMETERS_bay==1; parameters = [parameters; bay(:)]; end
    if LEARNPARAMETERS_Tau==1; parameters = [parameters; Tau(:)]; end
    parameters_keep = [];% if parameters_keep is 0 clamp the parameter to 0
    Whx_keep = ones(size(Whx)); Whh_keep = ones(size(Whh)); Wyh_keep = ones(size(Wyh)); bah_keep = ones(size(bah)); bay_keep = ones(size(bay)); Tau_keep = ones(size(Tau));
    Whh_keep(eye(numh,numh)==1) = 0;% no self connections
    if LEARNPARAMETERS_Whx==1; parameters_keep = [parameters_keep; Whx_keep(:)]; end
    if LEARNPARAMETERS_Whh==1; parameters_keep = [parameters_keep; Whh_keep(:)]; end
    if LEARNPARAMETERS_Wyh==1; parameters_keep = [parameters_keep; Wyh_keep(:)]; end
    if LEARNPARAMETERS_bah==1; parameters_keep = [parameters_keep; bah_keep(:)]; end
    if LEARNPARAMETERS_bay==1; parameters_keep = [parameters_keep; bay_keep(:)]; end
    if LEARNPARAMETERS_Tau==1; parameters_keep = [parameters_keep; Tau_keep(:)]; end
    

    %--------------------------------------------------------------------------
    %                      training and validation data
    %--------------------------------------------------------------------------
    % IN:         dimIN x numT x numtrain matrix
    % TARGETOUT:  dimOUT x numT x numtrain matrix
    % itimeRNN:   dimOUT x numT x numtrain matrix, elements 0(time-point does not contribute to first term in cost function), 1(time-point contributes to first term in cost function)
    randseed = epoch;
    [IN, TARGETOUT, itimeRNN] = generateINandTARGETOUT(dimIN,dimOUT,numT,numtrain,randseed,noiseamplitude_input,angle0duration,ANGULARVELOCITY,BOUNDARY);
    permuteIN = permute(IN,[1 3 2]);% dimIN x numtrain x numT matrix
    permuteitimeRNN = permute(itimeRNN,[1 3 2]);% dimOUT x numtrain x numT matrix
    randseedtest = epoch + 10000;
    [INtest, TARGETOUTtest, itimeRNNtest] = generateINandTARGETOUT(dimIN,dimOUT,numTtest,numtrialstest,randseedtest,noiseamplitude_input,angle0duration,ANGULARVELOCITY,BOUNDARY);   
    
    if 0
    figure;% target output on training trials
    %for itrial = 1:numtrain
    for itrial = numtrain
        clf; hold on; fontsize = 15; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',10,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)% only need to set DefaulttextFontSize if using function text
        T = [1:numT];
        A = TARGETOUT(:,:,itrial);% dimOUT x numT matrix
        plot([1:numT],IN(1,:,itrial),'k-');% for legend
        plot(T(itimeRNN(1,:,itrial)==1),A(1,itimeRNN(1,:,itrial)==1),'r--')% for legend
        
        plot([1:numT],IN(:,:,itrial),'k-')
        for i=1:dimOUT
            plot(T(itimeRNN(i,:,itrial)==1),A(i,itimeRNN(i,:,itrial)==1),'r--');
        end
        xlabel('Time steps')
        legend('Input','Target output','location','best');
        title(sprintf('Trial %g, %g time-steps in simulation',itrial,numT ));
        axis tight; axis([-Inf Inf min(ylim)-abs(max(ylim)-min(ylim))/100  max(ylim)+abs(max(ylim)-min(ylim))/100])
        set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
        %pause
    end
    end
    
    %--------------------------------------------------------------------------
    %                           model
    %               also update model on line 286 
    %--------------------------------------------------------------------------
    [Whx, Whh, Wyh, bah, bay, Tau] = unpack(parameters,dimIN,numh,dimOUT,Whx,Whh,Wyh,bah,bay,Tau,LEARNPARAMETERS_Whx,LEARNPARAMETERS_Whh,LEARNPARAMETERS_Wyh,LEARNPARAMETERS_bah,LEARNPARAMETERS_bay,LEARNPARAMETERS_Tau);
    [ah, h, ay, y, stuffforTau] = forwardpass(Whx,Whh,Wyh,bah,bay,Tau,ah0,h0,bahneverlearn,bhneverlearn,bayneverlearn,dt,IN,nonlinearity);
    h_withoutbias = computenonlinearity(ah,nonlinearity{1});% numh x numT x numexamples matrix
    permuteah = permute(ah,[1 3 2]);% numh x numtrain x numT matrix, originally ah is a numh x numT x numtrain matrix
    permuteh = permute(h,[1 3 2]);% numh x numtrain x numT matrix, originally h is a numh x numT x numtrain matrix
    permuteh_withoutbias = permute(h_withoutbias,[1 3 2]);% numh x numtrain x numT matrix
    permutey = permute(y,[1 3 2]);% dimOUT x numtrain x numT matrix, originally y is a dimOUT x numT x numtrain matrix
    
    %modelgradE.singleprecision = -700;% if model.singleprecision exists then variables in function are single precision, comment out line for double precision
    modelgradE.LEARNPARAMETERS_Whx = LEARNPARAMETERS_Whx;
    modelgradE.LEARNPARAMETERS_Whh = LEARNPARAMETERS_Whh;
    modelgradE.LEARNPARAMETERS_Wyh = LEARNPARAMETERS_Wyh;
    modelgradE.LEARNPARAMETERS_bah = LEARNPARAMETERS_bah;
    modelgradE.LEARNPARAMETERS_bay = LEARNPARAMETERS_bay;
    modelgradE.LEARNPARAMETERS_Tau = LEARNPARAMETERS_Tau;
    modelgradE.L1REGULARIZE_Whx = L1REGULARIZE_Whx;
    modelgradE.L1REGULARIZE_Whh = L1REGULARIZE_Whh;
    modelgradE.L1REGULARIZE_Wyh = L1REGULARIZE_Wyh;
    modelgradE.L1REGULARIZE_bah = L1REGULARIZE_bah;
    modelgradE.L1REGULARIZE_bay = L1REGULARIZE_bay;
    modelgradE.L1REGULARIZE_Tau = L1REGULARIZE_Tau;
    modelgradE.L2REGULARIZE_Whx = L2REGULARIZE_Whx;
    modelgradE.L2REGULARIZE_Whh = L2REGULARIZE_Whh;
    modelgradE.L2REGULARIZE_Wyh = L2REGULARIZE_Wyh;
    modelgradE.L2REGULARIZE_bah = L2REGULARIZE_bah;
    modelgradE.L2REGULARIZE_bay = L2REGULARIZE_bay;
    modelgradE.L2REGULARIZE_Tau = L2REGULARIZE_Tau;
    modelgradE.Whx = Whx;
    modelgradE.Whh = Whh;
    modelgradE.Wyh = Wyh;
    modelgradE.bah = bah;
    modelgradE.bay = bay;
    modelgradE.Tau = Tau;
    modelgradE.ah0 = ah0;
    modelgradE.h0 = h0;
    modelgradE.bahneverlearn = bahneverlearn;
    modelgradE.bhneverlearn = bhneverlearn;
    modelgradE.bayneverlearn = bayneverlearn;
    modelgradE.dt = dt;
    modelgradE.IN = IN;
    modelgradE.TARGETOUT = TARGETOUT;
    modelgradE.itimeRNN = itimeRNN;
    modelgradE.nonlinearity = nonlinearity;
    modelgradE.lambdaL1 = lambdaL1;
    modelgradE.lambdaL2 = lambdaL2;
    modelgradE.lambdahL1 = lambdahL1;
    modelgradE.lambdahL2 = lambdahL2;
    
    %modelgradEtest.singleprecision = -700;% if model.singleprecision exists then variables in function are single precision, comment out line for double precision
    modelgradEtest.LEARNPARAMETERS_Whx = LEARNPARAMETERS_Whx;
    modelgradEtest.LEARNPARAMETERS_Whh = LEARNPARAMETERS_Whh;
    modelgradEtest.LEARNPARAMETERS_Wyh = LEARNPARAMETERS_Wyh;
    modelgradEtest.LEARNPARAMETERS_bah = LEARNPARAMETERS_bah;
    modelgradEtest.LEARNPARAMETERS_bay = LEARNPARAMETERS_bay;
    modelgradEtest.LEARNPARAMETERS_Tau = LEARNPARAMETERS_Tau;
    modelgradEtest.L1REGULARIZE_Whx = L1REGULARIZE_Whx;
    modelgradEtest.L1REGULARIZE_Whh = L1REGULARIZE_Whh;
    modelgradEtest.L1REGULARIZE_Wyh = L1REGULARIZE_Wyh;
    modelgradEtest.L1REGULARIZE_bah = L1REGULARIZE_bah;
    modelgradEtest.L1REGULARIZE_bay = L1REGULARIZE_bay;
    modelgradEtest.L1REGULARIZE_Tau = L1REGULARIZE_Tau;
    modelgradEtest.L2REGULARIZE_Whx = L2REGULARIZE_Whx;
    modelgradEtest.L2REGULARIZE_Whh = L2REGULARIZE_Whh;
    modelgradEtest.L2REGULARIZE_Wyh = L2REGULARIZE_Wyh;
    modelgradEtest.L2REGULARIZE_bah = L2REGULARIZE_bah;
    modelgradEtest.L2REGULARIZE_bay = L2REGULARIZE_bay;
    modelgradEtest.L2REGULARIZE_Tau = L2REGULARIZE_Tau;
    modelgradEtest.Whx = Whx;
    modelgradEtest.Whh = Whh;
    modelgradEtest.Wyh = Wyh;
    modelgradEtest.bah = bah;
    modelgradEtest.bay = bay;
    modelgradEtest.Tau = Tau;
    modelgradEtest.ah0 = ah0test;
    modelgradEtest.h0 = h0test;
    modelgradEtest.bahneverlearn = bahneverlearntest;
    modelgradEtest.bhneverlearn = bhneverlearntest;
    modelgradEtest.bayneverlearn = bayneverlearntest;
    modelgradEtest.dt = dt;
    modelgradEtest.IN = INtest;
    modelgradEtest.TARGETOUT = TARGETOUTtest;
    modelgradEtest.itimeRNN = itimeRNNtest;
    modelgradEtest.nonlinearity = nonlinearity;
    modelgradEtest.lambdaL1 = lambdaL1;
    modelgradEtest.lambdaL2 = lambdaL2;
    modelgradEtest.lambdahL1 = lambdahL1;
    modelgradEtest.lambdahL2 = lambdahL2;

    %modelGv.singleprecision = -700;% if model.singleprecision exists then variables in function are single precision, comment out line for double precision
    modelGv.LEARNPARAMETERS_Whx = LEARNPARAMETERS_Whx;
    modelGv.LEARNPARAMETERS_Whh = LEARNPARAMETERS_Whh;
    modelGv.LEARNPARAMETERS_Wyh = LEARNPARAMETERS_Wyh;
    modelGv.LEARNPARAMETERS_bah = LEARNPARAMETERS_bah;
    modelGv.LEARNPARAMETERS_bay = LEARNPARAMETERS_bay;
    modelGv.LEARNPARAMETERS_Tau = LEARNPARAMETERS_Tau;
    modelGv.L2REGULARIZE_Whx = L2REGULARIZE_Whx;
    modelGv.L2REGULARIZE_Whh = L2REGULARIZE_Whh;
    modelGv.L2REGULARIZE_Wyh = L2REGULARIZE_Wyh;
    modelGv.L2REGULARIZE_bah = L2REGULARIZE_bah;
    modelGv.L2REGULARIZE_bay = L2REGULARIZE_bay;
    modelGv.L2REGULARIZE_Tau = L2REGULARIZE_Tau;
    modelGv.Whh = Whh;
    modelGv.Whx = Whx;
    modelGv.Wyh = Wyh;
    modelGv.bah = bah;% numh x 1 matrix
    modelGv.bay = bay;% dimOUT x 1 matrix
    modelGv.Tau = Tau;
    modelGv.ah0 = ah0(:,1:numtrainGv);
    modelGv.h0 = h0(:,1:numtrainGv);
    if isequal(bahneverlearn,0); modelGv.bahneverlearn = bahneverlearn; else; modelGv.bahneverlearn = bahneverlearn(:,:,1:numtrainGv); end
    modelGv.permuteIN = permuteIN(:,1:numtrainGv,:);
    modelGv.permuteah = permuteah(:,1:numtrainGv,:);
    modelGv.permuteh = permuteh(:,1:numtrainGv,:);
    modelGv.permuteh_withoutbias = permuteh_withoutbias(:,1:numtrainGv,:);
    modelGv.permutey = permutey(:,1:numtrainGv,:);
    modelGv.permuteitimeRNN = permuteitimeRNN(:,1:numtrainGv,:);
    modelGv.nonlinearity = nonlinearity;
    modelGv.dt = dt;
    modelGv.lambdaL2 = lambdaL2;
    modelGv.lambdaL2 = lambdahL2;
    modelGv.lambdaSD = lambdaSD;
    modelGv.stuffforTau = stuffforTau;
    
    %------------------------------------------------------------------------------------------------
    %                           gradient of error function 
    %                            use all training examples
    %------------------------------------------------------------------------------------------------
    tic
    [E, Emain, EL1, EL2, EhL1, EhL2, gradE] = computegradE(parameters,modelgradE);  
    if EL1 >= Emain/(2*3); lambdaL1 = lambdaL1*2/3; end% bound lambdaL1 from getting too large
    if EL1 <= Emain/(2*500); lambdaL1 = lambdaL1*3/2; end% bound lambdaL1 from getting too small
    if EL2 >= Emain/(2*3); lambdaL2 = lambdaL2*2/3; end% bound lambdaL2 from getting too large
    if EL2 <= Emain/(2*500); lambdaL2 = lambdaL2*3/2; end% bound lambdaL2 from getting too small
    if EhL1 >= Emain/(2*2); lambdahL1 = lambdahL1*2/3; end% bound lambdahL1 from getting too large
    if EhL1 <= Emain/(2*20); lambdahL1 = lambdahL1*3/2; end% bound lambdahL1 from getting too small
    if EhL2 >= Emain/(2*2); lambdahL2 = lambdahL2*2/3; end% bound lambdahL2 from getting too large
    if EhL2 <= Emain/(2*20); lambdahL2 = lambdahL2*3/2; end% bound lambdahL2 from getting too small
    lambdaSD = lambdaL2*mu;
    if DISPLAYON 
    disp(['Training epoch ' num2str(epoch)]); 
    disp(sprintf('lambdaL1 %g, lambdaL2 %g, lambdaSD %g, lambdahL1 %g, lambdahL2 %g',lambdaL1,lambdaL2,lambdaSD,lambdahL1,lambdahL2));
    end
    if LEARNPARAMETERS_Whh% information bottleneck, early in learning the CHANGE-in-weights is large and the fluctuations in the changes are small, later in learning the change in weights are small and the fluctuations in the change are large, https://www.youtube.com/watch?v=7VReK0DJLHQ&t=32m51s
        i = 1;
        %if LEARNPARAMETERS_Whx==1; gradEWhx = reshape(gradE(i:i+numh*dimIN-1),numh,dimIN); i = 1 + numh*dimIN; end
        if LEARNPARAMETERS_Whx==1; i = 1 + numh*dimIN; end
        if LEARNPARAMETERS_Whh==1; gradEWhh = gradE(i:i+numh*numh-1); i = i + numh*numh; end% numh*numh x 1 matrix
        %delta_gradEWhh = gradEWhh - gradEWhhprevious;% change in gradient
        %gradEWhhprevious = gradEWhh;
        meanabsgradEWhh_store(epoch) = mean(abs(gradEWhh));
        stdgradEWhh_store(epoch) = std(gradEWhh);
        if PLOTFIGURESDURINGTRAINING || epoch==numepoch
            hfigure5 = figure(5); clf;% meangradEWhh and stdgradEWhh for different training epochs
            subplot(3,1,1); hold on; fontsize = 12; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
            plot([1:epoch],meanabsgradEWhh_store(1:epoch),'k-')
            ylabel('mean(abs(gradEWhh))')
            title({[sprintf('Epoch %g, mean(abs(gradEWhh)) = %g, std(gradEWhh) = %g',epoch,meanabsgradEWhh_store(epoch),stdgradEWhh_store(epoch))];[sprintf('mean(abs(gradEWhh)): Min = %g, Mean = %g, Max = %g',min(meanabsgradEWhh_store(1:epoch)),mean(meanabsgradEWhh_store(1:epoch)),max(meanabsgradEWhh_store(1:epoch)))];[sprintf('std(gradEWhh): Min = %g, Mean = %g, Max = %g',min(stdgradEWhh_store(1:epoch)),mean(stdgradEWhh_store(1:epoch)),max(stdgradEWhh_store(1:epoch)))]})
            subplot(3,1,2); hold on; fontsize = 12; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
            plot([1:epoch],stdgradEWhh_store(1:epoch),'r-')
            ylabel('std(gradEWhh)')
            subplot(3,1,3); fontsize = 12; set(gcf,'DefaultLineLineWidth',1,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
            semilogy([1:epoch],meanabsgradEWhh_store(1:epoch),'k-'); hold on;
            semilogy([1:epoch],stdgradEWhh_store(1:epoch),'r-')
            legend('mean(abs(gradEWhh))','std(gradEWhh)','location','best')
            xlabel('Training epoch')
            set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
            if epoch==numepoch || rem(epoch,5)==0; print(hfigure5, '-dpdf', sprintf('%s/gradEWhhVStrainingepoch',figuredir)); end 
        end
    end

    %---------------
    % update model 
    modelgradE.lambdaL1 = lambdaL1;
    modelgradE.lambdaL2 = lambdaL2;
    modelgradE.lambdahL1 = lambdahL1;
    modelgradE.lambdahL2 = lambdahL2;
    modelgradEtest.lambdaL1 = lambdaL1;
    modelgradEtest.lambdaL2 = lambdaL2;
    modelgradEtest.lambdahL1 = lambdahL1;
    modelgradEtest.lambdahL2 = lambdahL2;
    modelGv.lambdaL2 = lambdaL2;
    modelGv.lambdaSD = lambdaSD;
    modelGv.lambdahL2 = lambdahL2;

    
    maxnormgradE = 10;% clip gradients to a maximum norm value of maxnormgradE, Pascanu et al. 2013 "On the difficulty of training recurrent neural networks"
    normgradE = sqrt( gradE' * gradE );
    normgradE_store(epoch) = normgradE;
    if PLOTFIGURESDURINGTRAINING || epoch==numepoch
    hfigure4 = figure(4);% normgradE for different epochs
    clf; hold on; fontsize = 15; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize) 
    plot([1:epoch],normgradE_store(1:epoch),'k-')
    xlabel('Training epoch'); ylabel('norm(gradE)')
    title({[sprintf('Epoch %g, norm(gradE) = %g',epoch,normgradE)];[sprintf('Min = %g, Mean = %g, Max = %g',min(normgradE_store(1:epoch)),mean(normgradE_store(1:epoch)),max(normgradE_store(1:epoch)))]})
    set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
    if epoch==numepoch || rem(epoch,5)==0; warning('off', 'MATLAB:print:FileName'); print(hfigure4, '-dpdf', sprintf('%s/normgradE',figuredir)); end% turn off warning about figure name being the same as variable name 
    end
    if normgradE > maxnormgradE; gradE = (maxnormgradE/normgradE) * gradE; normgradE = sqrt( gradE' * gradE ); end% clip gradients to a maximum norm value of maxnormgradE, Pascanu et al. 2013 "On the difficulty of training recurrent neural networks"
    %------------------------------------------------------------------------------------------------
    %                           conjugate gradient algorithm 
    %             to minimize positive-definite quadratic approximation of the error function
    %------------------------------------------------------------------------------------------------
    %parametersCG = 0.95*parametersCG;% slightly decay the previous parameter vector before using it to initialize CG
    if rem(epoch,10)==0% initialize parametersCG with gradient descent step
        normparametersCG = sqrt(parametersCG' * parametersCG);
        gradErescaled = (2*normparametersCG/normgradE) * gradE;% numparameters x 1 matrix, norm of gradErescaled is 10 times the norm of parametersCG
        parametersCG = -gradErescaled;
    end
    if rem(epoch,10)~=0
        parametersCG = 0.95*parametersCG;% slightly decay the previous parameter vector before using it to initialize CG
    end
    normparametersCG = sqrt(parametersCG' * parametersCG); if DISPLAYON; display(sprintf('normparametersCGinit = %g',normparametersCG)); end
    maxiters = 50;% "maxiters is the most important variable that you should try tweaking"
    miniters = 1;
    precon = eye(numparameters,numparameters); 
        
    
    [parametersCGstore, istore] = conjgrad( @(v)computeGv(v,modelGv), -gradE, parametersCG, maxiters, miniters, precon ); 
    for iii=1:size(parametersCGstore,2)% set to 0 parameters we want to be 0 to enforce input and output from subpopulations of RNN
        parametersCGstore(parameters_keep==0,iii) = 0;
    end


    %---------------------------------------------------------------------------------------------------  
    %                          CG-backtracking
    %                     using the full training set
    %---------------------------------------------------------------------------------------------------  
    xx = parameters + parametersCGstore(:,end);% numparameters x 1 matrix
    error = computegradE(xx,modelgradE);
    
    for j = length(istore)-1:-1:1
        xx = parameters + parametersCGstore(:,j);% numparameters x 1 matrix
        error_new = computegradE(xx,modelgradE);
        if error < error_new
            j = j + 1;
            break;
        end
        error = error_new;
    end
    if isempty(j)% if length(istore) is 1
        j = 1;
    end
    parametersCG = parametersCGstore(:,j);
    %display([num2str(istore(j)) 'out of ' num2str(maxiters) ' CG steps used']);
    %-----------------------------------------------------------------------------------------------
    %                              compute reduction ratio rho
    %                    rho = (error(theta+dtheta) - error(theta)) / (error_localquadraticapproximation(dtheta) - error_localquadraticapproximation(0))
    %-----------------------------------------------------------------------------------------------
    xx = parameters;% numparameters x 1 matrix
    error_old = computegradE(xx,modelgradE);
    
    xx = parameters + parametersCG;% numparameters x 1 matrix
    error_new = computegradE(xx,modelgradE);
    
    Enew_minus_Eold_quadraticapproximation = 0.5*(parametersCG'*computeGv(parametersCG,modelGv)) + gradE'*parametersCG;% a negative number, Whx, Whh, etc. are the old parameters
    rho = (error_new - error_old) / Enew_minus_Eold_quadraticapproximation;
    if error_new > error_old
        rho = -Inf;
    end
    rho_store(epoch) = rho;
    % Martens and Sutskever 2012 
    % rho measures the ratio of the reduction in the error function E(theta_t-1 + epsilon) - E(theta_t-1) produced bay the update epsilon, 
    % to the amount of reduction predicted bay the quadratic model. 
    % When rho is much smaller than 1, the quadratic model overestimates the amount of reduction 
    % and so lambdaL2 should be increased, encouraging future updates to be more conservative and thus lie somewhere that the quadratic model more accurately predicts the reduction. 
    % In contrast, when rho is close to 1, the quadratic approximation is likely to be fairly accurate near the parameter value theta, and so we can afford to reduce lambdaL2, thus relaxing the constraints on theta and allowing for "larger" and more substantial updates.
    
    %------------------------------------------------------------------------------------------------
    %                                  backtracking line search 
    %                          to find a scalar (rate) to multiply the parameter update (parametersCG)
    %------------------------------------------------------------------------------------------------
    rate = 1;
    c = 10^(-2);
    j = 0;
    while j < 60
        if error_new <= error_old + c * rate * (gradE' * parametersCG)
            break;
        else
            rate = 0.8*rate;
            j = j + 1;
        end
        
        xx = parameters + rate*parametersCG;% numparameters x 1 matrix
        error_new = computegradE(xx,modelgradE);
    end
    
    % reject the step
    if j == 60
        j = Inf;
        rate = 0;
        error_new = error_old;
    end
    
    %------------------------------------------------------------------------------------------------
    %                           update damping parameter lambdaL2 
    %                            Levenberg-Marquardt heuristic
    %------------------------------------------------------------------------------------------------
    if rho > 3/4
        lambdaL2 = lambdaL2 * 2/3;
    elseif rho < 1/4
        lambdaL2 = lambdaL2 * 3/2;
    end
    
    
    %------------------------------------------------------------------------------------------------
    %                           update the parameters
    %------------------------------------------------------------------------------------------------
    tictoctime_store(epoch) = toc;
    parameters = parameters + rate*parametersCG;
    [Whx, Whh, Wyh, bah, bay, Tau] = unpack(parameters,dimIN,numh,dimOUT,Whx,Whh,Wyh,bah,bay,Tau,LEARNPARAMETERS_Whx,LEARNPARAMETERS_Whh,LEARNPARAMETERS_Wyh,LEARNPARAMETERS_bah,LEARNPARAMETERS_bay,LEARNPARAMETERS_Tau);
    if SAVE
        if ismember(epoch,epochset_saveparameters); eval(sprintf('parameters_saveepoch%g = [ah0(:); h0(:); Whx(:); Whh(:); Wyh(:); bah; bay; Tau];',epoch)); save(sprintf('parameters_saveepoch%g.mat',epoch), sprintf('parameters_saveepoch%g',epoch)); end
    end
    normdeltaparameters_store(epoch) = sqrt(rate*parametersCG' * rate*parametersCG);
    [error, errormain, errorL1, errorL2, errorhL1, errorhL2] = computegradE(parameters,modelgradE);
    [ah, h, ay, y] = forwardpass(Whx,Whh,Wyh,bah,bay,Tau,ah0,h0,bahneverlearn,bhneverlearn,bayneverlearn,dt,IN,nonlinearity);
    medianabsah_store(epoch) = median(abs(ah(:)));
    meanabsah_store(epoch) = mean(abs(ah(:)));
    maxabsah_store(epoch) = max(abs(ah(:)));
    maxabsparameters_store(epoch) = max(abs(parameters));
    error_store(epoch) = error; 
    errormain_store(epoch) = errormain; 
    errorL1_store(epoch) = errorL1;
    errorL2_store(epoch) = errorL2;
    errorhL1_store(epoch) = errorhL1;
    errorhL2_store(epoch) = errorhL2;
    %--------------------------------
    % calculate errormain on test set
    [~, errormaintest] = computegradE(parameters,modelgradEtest); 
    errormaintest_store(epoch) = errormaintest;
    
    
    if PLOTFIGURESDURINGTRAINING || epoch==numepoch
    hfigure1 = figure(1); clf; 
    itrial = (1*(rem(epoch,2)==1) + numtrain*(rem(epoch,2)==0));% alternate between trial 1 and numtrain
    Tplot = [1:numT];
    subplot(2,1,1); hold on; fontsize = 10; set(gcf,'DefaultLineLineWidth',1,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',0.5); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize,'FontWeight','normal');
    A = TARGETOUT(:,Tplot,itrial);% dimOUT x numT matrix
    colors = parula(dimOUT);% colors(1,:) is purple, blue, green, colors(end,:) is yellow 
    if dimOUT==1; colors = [1 0 0]; end% if dimOUT==1 then RNN output is red 
    plot(Tplot(itimeRNN(1,Tplot,itrial)==1),A(1,itimeRNN(1,Tplot,itrial)==1),'k-','linewidth',1); plot(Tplot,y(end,Tplot,itrial),'color',colors(end,:),'linestyle','-','linewidth',1)% for legend
    for i=1:dimOUT
       plot(Tplot(itimeRNN(i,Tplot,itrial)==1),A(i,itimeRNN(i,Tplot,itrial)==1),'k-','linewidth',1);% target output included in first term of error function 
    end
    for i=1:dimOUT
        plot(Tplot,y(i,Tplot,itrial),'color',colors(i,:),'linestyle','-','linewidth',1)% RNN output
    end
    legend('Target output','RNN output','location','best')
    title(sprintf('Epoch %g, error %g, median(abs(ah(:))) = %g, mean(abs(ah(:))) = %g, max(abs(ah(:))) = %g',epoch,error,median(abs(ah(:))),mean(abs(ah(:))),max(abs(ah(:)))))
    set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
    axis tight
    subplot(2,1,2); hold on; fontsize = 10; set(gcf,'DefaultLineLineWidth',1,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',0.5); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize,'FontWeight','normal');
    colors = parula(dimIN);% colors(1,:) is purple, blue, green, colors(end,:) is yellow 
    if dimIN==1; colors = [0 0 0]; end% if dimIN==1 then input is black 
    plot(Tplot,IN(end,Tplot,itrial),'color',colors(end,:),'linestyle','-','linewidth',1); plot(Tplot,h(1,Tplot,itrial),'r-','linewidth',1);% for legend
    plot(Tplot,h(:,Tplot,itrial),'r-','linewidth',1)
    for i=1:dimIN
        plot(Tplot,IN(i,Tplot,itrial),'color',colors(i,:),'linestyle','-','linewidth',1)
    end
    plot(Tplot,mean(h(:,Tplot,itrial),1),'b--','linewidth',1)
    xlabel('Time steps')
    legend('Input','Hidden units','location','best')
    set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
    axis tight
    movegui(hfigure1,'northwest'); drawnow;
    if epoch==numepoch || rem(epoch,5)==0; print(hfigure1, '-dpdf', sprintf('%s/traintrial%g',figuredir,itrial)); end
    end
    
    
    if PLOTFIGURESDURINGTRAINING || epoch==numepoch
    hfigure2 = figure(2); clf; hold on;
    subplot(2,1,1); hold on; fontsize = 10; set(gcf,'DefaultLineLineWidth',1,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',0.5); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize,'FontWeight','normal');
    legendstring = [];
    ii = 1;
    if LEARNPARAMETERS_Whx==1; legendstring = [legendstring; 'Whx']; plot(1:1+numh*dimIN-1,Whx(:),'r-'); ii = 1 + numh*dimIN; end% Whx
    if LEARNPARAMETERS_Whh==1; legendstring = [legendstring; 'Whh']; plot(ii:ii+numh*numh-1,Whh(:),'k-'); ii = ii + numh*numh; end% Whh
    if LEARNPARAMETERS_Wyh==1; legendstring = [legendstring; 'Wyh']; plot(ii:ii+dimOUT*numh-1,Wyh(:),'g-'); ii = ii + dimOUT*numh; end% Wyh
    if LEARNPARAMETERS_bah==1; legendstring = [legendstring; 'bah']; plot(ii:ii+numh-1,bah,'b-'); ii = ii + numh; end% bah
    if LEARNPARAMETERS_bay==1; legendstring = [legendstring; 'bay']; plot(ii:ii+dimOUT-1,bay,'m-'); ii = ii + dimOUT; end% bay
    if LEARNPARAMETERS_Tau==1; legendstring = [legendstring; 'Tau']; plot(ii:ii+numh-1,Tau,'c-'); ii = ii + numh; end% Tau
    xlabel('Parameter number'); ylabel('Parameter value')
    legend(legendstring,'location','best');
    title(sprintf('Epoch %g, error %g, median(abs(ah(:))) = %g, mean(abs(ah(:))) = %g, max(abs(ah(:))) = %g',epoch,error,median(abs(ah(:))),mean(abs(ah(:))),max(abs(ah(:)))))
    set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
    axis tight
    subplot(2,1,2); hold on; fontsize = 10; set(gcf,'DefaultLineLineWidth',1,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',0.5); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize,'FontWeight','normal');
    plot([1:epoch],maxabsparameters_store(1:epoch),'k-')
    plot([1 ceil(epoch/4) round(epoch/2) epoch],maxabsparameters_store([1 ceil(epoch/4) round(epoch/2) epoch]),'ko')
    xlabel('Training epoch'); ylabel('max(abs(parameters))')
    title(sprintf('max(abs(parameters)) epoch %g = %.4g, epoch %g = %.4g, epoch %g = %.4g, epoch %g = %.4g',1,maxabsparameters_store(1),ceil(epoch/4),maxabsparameters_store(ceil(epoch/4)),round(epoch/2),maxabsparameters_store(round(epoch/2)),epoch,maxabsparameters_store(epoch))) 
    set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
    axis tight
    movegui(hfigure2,'northeast'); drawnow;    
    if epoch==numepoch || rem(epoch,5)==0; warning('off', 'MATLAB:print:FileName'); print(hfigure2, '-dpdf', sprintf('%s/parameters',figuredir)); end
    end
    
    if PLOTFIGURESDURINGTRAINING || epoch==numepoch  
    hfigure3 = figure(3); clf;
    if error_store(epoch)>0
    semilogy([1:epoch],error_store(1:epoch),'k-'); hold on;
    semilogy([1:epoch],errormain_store(1:epoch),'r-');
    semilogy([1:epoch],errorL1_store(1:epoch),'y-');
    semilogy([1:epoch],errorL2_store(1:epoch),'g-');
    semilogy([1:epoch],errorhL1_store(1:epoch),'c-');
    semilogy([1:epoch],errorhL2_store(1:epoch),'m-');
    semilogy([1:epoch],errormaintest_store(1:epoch),'b--');
    end
    if error_store(epoch)<0
    plot([1:epoch],error_store(1:epoch),'k-'); hold on;
    plot([1:epoch],errormain_store(1:epoch),'r-');
    plot([1:epoch],errorL1_store(1:epoch),'y-');
    plot([1:epoch],errorL2_store(1:epoch),'g-');
    semilogy([1:epoch],errorhL1_store(1:epoch),'c-');
    semilogy([1:epoch],errorhL2_store(1:epoch),'m-');
    plot([1:epoch],errormaintest_store(1:epoch),'b--');
    end
    legend('Error','Error main','Error L1','Error L2','Error hL1','Error hL2','Error main test','location','best')
    xlabel('Training epoch'); ylabel('Error')
    title(sprintf('Epoch %.4g, error %.4g, errormain %.4g, errorL1 %.4g, errorL2 %.4g, errorhL1 %.4g, errorhL2 %.4g, errormaintest %.4g',epoch,error,errormain,errorL1,errorL2,errorhL1,errorhL2,errormaintest))
    set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
    movegui(hfigure3,'north'); drawnow;
    if epoch==numepoch || rem(epoch,5)==0; print(hfigure3, '-dpdf', sprintf('%s/error_train_',figuredir)); end
    end
    if DISPLAYON
    display(sprintf('normdeltaparameters = %g, normgradE before clipping = %g, maxnormgradE = %g',normdeltaparameters_store(epoch),normgradE_store(epoch),maxnormgradE)); 
    disp(['rho: ' num2str(rho)]); 
    disp(sprintf('errormaintest %g, errormain %g, errorL1 %g, errorL2 %g, errorhL1 %g, errorhL2 %g, error %g',errormaintest,errormain,errorL1,errorL2,errorhL1,errorhL2,error)); 
    %if strcmp(nonlinearity{2},'tanh'); iscorrect = -700*ones(numtrain,1); for itrial=1:numtrain; iscorrect(itrial) = (  sign(y(1,itimeRNN(1,:,itrial)==1,itrial)) == sign(TARGETOUT(1,itimeRNN(1,:,itrial)==1,itrial))  ); end; end% when output is discrete calculate percent correct, targetout in {-0.5,0.5}
    %if strcmp(nonlinearity{2},'logistic'); iscorrect = -700*ones(numtrain,1); for itrial=1:numtrain; iscorrect(itrial) = ((y(1,itimeRNN(1,:,itrial)==1,itrial)>=0.5) == TARGETOUT(1,itimeRNN(1,:,itrial)==1,itrial)); end; end% when output is discrete calculate percent correct, targetout in {0,1} 
    %disp(sprintf('percent correct = %g',100*sum(iscorrect)/numtrain))  
    disp('====================='); 
    end
end% for epoch=1:numepoch
if SAVE
save('tictoctime_store.mat','tictoctime_store')     
save('error_store.mat','error_store')
save('errormain_store.mat','errormain_store')
save('errorL1_store.mat','errorL1_store')
save('errorL2_store.mat','errorL2_store')
save('errorhL1_store.mat','errorhL1_store')
save('errorhL2_store.mat','errorhL2_store')
save('normgradE_store.mat','normgradE_store')
save('normdeltaparameters_store.mat','normdeltaparameters_store')
save('medianabsah_store.mat','medianabsah_store')
save('meanabsah_store.mat','meanabsah_store')
save('maxabsah_store.mat','maxabsah_store')
end


% numepoch = 25;
% tictoctime_store(numepoch+1:end) = [];
% error_store(numepoch+1:end) = [];
% errormain_store(numepoch+1:end) = [];
% errorL1_store(numepoch+1:end) = [];
% errorL2_store(numepoch+1:end) = [];
% errorhL1_store(numepoch+1:end) = [];
% errorhL2_store(numepoch+1:end) = [];
% normgradE_store(numepoch+1:end) = [];
% normdeltaparameters_store(numepoch+1:end) = [];
% medianabsah_store(numepoch+1:end) = [];
% meanabsah_store(numepoch+1:end) = [];
% maxabsah_store(numepoch+1:end) = [];
% % 
% datadir = '/Users/ccueva1/Desktop/neural networks/Shadlen/Naomi/simulation3_uniform';
% load(fullfile(datadir,'tictoctime_store.mat'))
% load(fullfile(datadir,'error_store.mat'))
% load(fullfile(datadir,'errormain_store.mat'))
% load(fullfile(datadir,'errorL1_store.mat'))
% load(fullfile(datadir,'errorL2_store.mat'))
% load(fullfile(datadir,'errorhL1_store.mat'))
% load(fullfile(datadir,'errorhL2_store.mat'))
% load(fullfile(datadir,'normgradE_store.mat'))
% load(fullfile(datadir,'normdeltaparameters_store.mat'))
% load(fullfile(datadir,'medianabsah_store.mat'))
% load(fullfile(datadir,'meanabsah_store.mat'))
% load(fullfile(datadir,'maxabsah_store.mat'))
handle = figure;
fontsize = 12; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize) 
if min(error_store)>0
    semilogy([1:numepoch],error_store,'k-'); hold on;
    semilogy([1:numepoch],errormain_store,'r-');
    semilogy([1:numepoch],errorL1_store,'y-');
    semilogy([1:numepoch],errorL2_store,'g-');
    semilogy([1:numepoch],errorhL1_store,'c-');
    semilogy([1:numepoch],errorhL2_store,'m-');
    semilogy(round([5 numepoch/4 numepoch/2 numepoch]),error_store(round([5 numepoch/4 numepoch/2 numepoch])),'ko')
    semilogy(round([5 numepoch/4 numepoch/2 numepoch]),errormain_store(round([5 numepoch/4 numepoch/2 numepoch])),'ro')
    semilogy(round([5 numepoch/4 numepoch/2 numepoch]),errorL1_store(round([5 numepoch/4 numepoch/2 numepoch])),'yo')
    semilogy(round([5 numepoch/4 numepoch/2 numepoch]),errorL2_store(round([5 numepoch/4 numepoch/2 numepoch])),'go')
    semilogy(round([5 numepoch/4 numepoch/2 numepoch]),errorhL1_store(round([5 numepoch/4 numepoch/2 numepoch])),'co')
    semilogy(round([5 numepoch/4 numepoch/2 numepoch]),errorhL2_store(round([5 numepoch/4 numepoch/2 numepoch])),'mo')
end
if min(error_store)<=0
    plot([1:numepoch],error_store,'k-'); hold on;
    plot([1:numepoch],errormain_store,'r-');
    plot([1:numepoch],errorL1_store,'y-');
    plot([1:numepoch],errorL2_store,'g-');
    plot([1:numepoch],errorhL1_store,'c-');
    plot([1:numepoch],errorhL2_store,'m-');
    plot(round([5 numepoch/4 numepoch/2 numepoch]),error_store(round([5 numepoch/4 numepoch/2 numepoch])),'ko')
    plot(round([5 numepoch/4 numepoch/2 numepoch]),errormain_store(round([5 numepoch/4 numepoch/2 numepoch])),'ro')
    plot(round([5 numepoch/4 numepoch/2 numepoch]),errorL1_store(round([5 numepoch/4 numepoch/2 numepoch])),'yo')
    plot(round([5 numepoch/4 numepoch/2 numepoch]),errorL2_store(round([5 numepoch/4 numepoch/2 numepoch])),'go')
    plot(round([5 numepoch/4 numepoch/2 numepoch]),errorhL1_store(round([5 numepoch/4 numepoch/2 numepoch])),'co')
    plot(round([5 numepoch/4 numepoch/2 numepoch]),errorhL2_store(round([5 numepoch/4 numepoch/2 numepoch])),'mo')
end
xlabel('Training epoch'); ylabel('Error')
legend('Total error','errormain','error L1','error L2','error hL1','error hL2','location','best'); legend boxoff
title({[sprintf('Error epoch %g  = %.4g, epoch %g = %.4g, epoch %g = %.4g, epoch %g = %.4g',5,error_store(5),round(numepoch/4),error_store(round(numepoch/4)),round(numepoch/2),error_store(round(numepoch/2)),numepoch,error_store(numepoch))];...
       [sprintf('Errormain epoch %g  = %.4g, epoch %g = %.4g, epoch %g = %.4g, epoch %g = %.4g',5,errormain_store(5),round(numepoch/4),errormain_store(round(numepoch/4)),round(numepoch/2),errormain_store(round(numepoch/2)),numepoch,errormain_store(numepoch))];...
       [sprintf('ErrorL1 epoch %g  = %.4g, epoch %g = %.4g, epoch %g = %.4g, epoch %g = %.4g',5,errorL1_store(5),round(numepoch/4),errorL1_store(round(numepoch/4)),round(numepoch/2),errorL1_store(round(numepoch/2)),numepoch,errorL1_store(numepoch))];...
       [sprintf('ErrorL2 epoch %g  = %.4g, epoch %g = %.4g, epoch %g = %.4g, epoch %g = %.4g',5,errorL2_store(5),round(numepoch/4),errorL2_store(round(numepoch/4)),round(numepoch/2),errorL2_store(round(numepoch/2)),numepoch,errorL2_store(numepoch))];...
       [sprintf('ErrorhL1 epoch %g  = %.4g, epoch %g = %.4g, epoch %g = %.4g, epoch %g = %.4g',5,errorhL1_store(5),round(numepoch/4),errorhL1_store(round(numepoch/4)),round(numepoch/2),errorhL1_store(round(numepoch/2)),numepoch,errorhL1_store(numepoch))];...
       [sprintf('ErrorhL2 epoch %g  = %.4g, epoch %g = %.4g, epoch %g = %.4g, epoch %g = %.4g',5,errorhL2_store(5),round(numepoch/4),errorhL2_store(round(numepoch/4)),round(numepoch/2),errorhL2_store(round(numepoch/2)),numepoch,errorhL2_store(numepoch))]})
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
print(handle, '-dpdf', sprintf('%s/error_train',figuredir))    
   
handle = figure;
hold on; fontsize = 12; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize) 
plot([1:numepoch],maxabsah_store,'r-')
plot([1:numepoch],meanabsah_store,'k-')
plot([1:numepoch],medianabsah_store,'g-')
plot(round([5 numepoch/4 numepoch/2 numepoch]),maxabsah_store(round([5 numepoch/4 numepoch/2 numepoch])),'ro')
plot(round([5 numepoch/4 numepoch/2 numepoch]),meanabsah_store(round([5 numepoch/4 numepoch/2 numepoch])),'ko')
plot(round([5 numepoch/4 numepoch/2 numepoch]),medianabsah_store(round([5 numepoch/4 numepoch/2 numepoch])),'go')
legend('max(abs(ah(:)))','mean(abs(ah(:)))','median(abs(ah(:)))','location','best'); legend boxoff
xlabel('Training epoch')
title({[sprintf('max(abs(ah(:))) epoch %g  = %.4g, epoch %g = %.4g, epoch %g = %.4g, epoch %g = %.4g',5,maxabsah_store(5),round(numepoch/4),maxabsah_store(round(numepoch/4)),round(numepoch/2),maxabsah_store(round(numepoch/2)),numepoch,maxabsah_store(numepoch))];...
       [sprintf('mean(abs(ah(:))) epoch %g  = %.4g, epoch %g = %.4g, epoch %g = %.4g, epoch %g = %.4g',5,meanabsah_store(5),round(numepoch/4),meanabsah_store(round(numepoch/4)),round(numepoch/2),meanabsah_store(round(numepoch/2)),numepoch,meanabsah_store(numepoch))];...
       [sprintf('median(abs(ah(:))) epoch %g  = %.4g, epoch %g = %.4g, epoch %g = %.4g, epoch %g = %.4g',5,medianabsah_store(5),round(numepoch/4),medianabsah_store(round(numepoch/4)),round(numepoch/2),medianabsah_store(round(numepoch/2)),numepoch,medianabsah_store(numepoch))]})
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
warning('off', 'MATLAB:print:FileName'); print(handle, '-dpdf', sprintf('%s/ah',figuredir));% turn off warning about figure name being the same as variable name 

handle = figure;% normgradE and normdeltaparameters versus training epoch
hold on; fontsize = 15; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize) 
plot([1:numepoch],normgradE_store / max(normgradE_store(:)),'k-')
plot([1:numepoch],normdeltaparameters_store / max(normdeltaparameters_store(:)),'r-')
legend('Scaled normgradE','Scaled normdeltaparameters','location','best'); legend boxoff
xlabel('Training epoch')
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
print(handle, '-dpdf', sprintf('%s/normgradE_normdeltaparameters',figuredir));



























%%
if isfield(ANGULARVELOCITY,'discreteangularvelocitydegrees')% if RNN was trained using discrete angular inputs, test RNN with continuous angular inputs
    ANGULARVELOCITY = rmfield(ANGULARVELOCITY,'discreteangularvelocitydegrees');
    %ANGULARVELOCITY.discreteangularvelocitydegrees = [-8 0 0 0 8];% if this variable exists then the angular velocity is drawn randomly from this discrete set at each timestep, and angularmomentum and sd are not used
    ANGULARVELOCITY.angularmomentum = 0.8;% angularvelocity(t) = sd*randn + angularmomentum * angularvelocity(t-1), angularmomentum and sd are not used when using ANGULARVELOCITY.discreteangularvelocitydegrees
    ANGULARVELOCITY.sd = ANGULARSPEED * 0.03;% at each timestep the new angularvelocity is a gaussian random variable with mean 0 and standard deviation sd, angularmomentum and sd are not used when using ANGULARVELOCITY.discreteangularvelocitydegrees
end 


%ANGULARSPEED = 0.1;% test with ANGULARSPEED0DOT1
%ANGULARSPEED = 3;% test with ANGULARSPEED3
%ANGULARVELOCITY.sd = ANGULARSPEED * 0.03;% at each timestep the new angularvelocity is a gaussian random variable with mean 0 and standard deviation sd, angularmomentum and sd are not used when using ANGULARVELOCITY.discreteangularvelocitydegrees

%--------------------------------------------------------------------------
%                          Test data
%--------------------------------------------------------------------------
% numtrain > numtrials, numT > numTtest, IN > INtest, TARGETOUT > OUTtest
numtrials = 2000;% 5000
numTtest = 1000;% 1000
%numTtest = 200;
randseedtest = 99999999;
[INtest, TARGETOUTtest, itimeRNNtest, angle_radians, angularvelocity_store] = generateINandTARGETOUT(dimIN,dimOUT,numTtest,numtrials,randseedtest,noiseamplitude_input,angle0duration,ANGULARVELOCITY,BOUNDARY);
% IN:         dimIN x numT x numtrials matrix
% TARGETOUT:  dimOUT x numT x numtrials matrix
% itimeRNN:   dimOUT x numT x numtrials matrix, elements 0(time-point does not contribute to first term in cost function), 1(time-point contributes to first term in cost function)
%bahneverlearntest = randn(numh,numTtest,numtrials);
%bahneverlearntest = zeros(numh,numTtest,numtrials);
%bhneverlearntest = zeros(numh,numTtest,numtrials);
%bayneverlearntest = zeros(dimOUT,numTtest,numtrials);   
bahneverlearntest = 0; bhneverlearntest = 0; bayneverlearntest = 0;% set to 0 instead of zeros(...,numT,numexamples) to save memory and run faster, code is written so this only works if all never-learned biases are 0
modelgradEtest.LEARNPARAMETERS_Whx = 0;
modelgradEtest.LEARNPARAMETERS_Whh = 0;
modelgradEtest.LEARNPARAMETERS_Wyh = 0;
modelgradEtest.LEARNPARAMETERS_bah = 0;
modelgradEtest.LEARNPARAMETERS_bay = 0;
modelgradEtest.LEARNPARAMETERS_Tau = 0;
modelgradEtest.L1REGULARIZE_Whx = 0;
modelgradEtest.L1REGULARIZE_Whh = 0;
modelgradEtest.L1REGULARIZE_Wyh = 0;
modelgradEtest.L1REGULARIZE_bah = 0;
modelgradEtest.L1REGULARIZE_bay = 0;
modelgradEtest.L1REGULARIZE_Tau = 0;
modelgradEtest.L2REGULARIZE_Whx = 0;
modelgradEtest.L2REGULARIZE_Whh = 0;
modelgradEtest.L2REGULARIZE_Wyh = 0;
modelgradEtest.L2REGULARIZE_bah = 0;
modelgradEtest.L2REGULARIZE_bay = 0;
modelgradEtest.L2REGULARIZE_Tau = 0;
modelgradEtest.dt = dt;
modelgradEtest.IN = INtest;
modelgradEtest.TARGETOUT = TARGETOUTtest;
modelgradEtest.itimeRNN = itimeRNNtest;
modelgradEtest.nonlinearity = nonlinearity;
modelgradEtest.lambdaL1 = 0;
modelgradEtest.lambdaL2 = 0;
modelgradEtest.lambdahL1 = 0;
modelgradEtest.lambdahL2 = 0;
modelgradEtest.bahneverlearn = bahneverlearntest;
modelgradEtest.bhneverlearn = bhneverlearntest;
modelgradEtest.bayneverlearn = bayneverlearntest;

%--------------------------------------------------------------------------
%              errormain as a function of training epoch
%--------------------------------------------------------------------------
epochset = epochset_saveparameters(epochset_saveparameters>=5);% epochset = [5 10 15...490 495 500 600 700...numepoch]
error_store = -700*ones(size(epochset));
errornormalized_store = -700*ones(size(epochset));
for i=1:length(epochset)
    epoch = epochset(i);
    if exist(fullfile(datadir,sprintf('parameters_saveepoch%g.mat',epoch)),'file')==2
        load(fullfile(datadir,sprintf('parameters_saveepoch%g.mat',epoch)));
        eval(sprintf('parameterssave = parameters_saveepoch%g;',epoch));
        [ah0, h0, Whx, Whh, Wyh, bah, bay, Tau] = unpackall(parameterssave,dimIN,numh,dimOUT,numtrain);
        modelgradEtest.ah0 = ah0(:,1)*ones(1,numtrials);
        modelgradEtest.h0 = h0(:,1)*ones(1,numtrials);
        modelgradEtest.Whx = Whx;
        modelgradEtest.Whh = Whh;
        modelgradEtest.Wyh = Wyh;
        modelgradEtest.bah = bah;
        modelgradEtest.bay = bay;
        modelgradEtest.Tau = Tau;
        [~, errormain] = computegradE([],modelgradEtest);
        error_store(i) = errormain;
        
        %---------------------------
        % normalized error, if RNN output is constant for each dimOUT (each dimOUT can be a different constant) then normalizederror = 100%
        ah0test = ah0(:,1)*ones(1,numtrials);
        h0test = h0(:,1)*ones(1,numtrials);
        [ah, h, ay, y] = forwardpass(Whx,Whh,Wyh,bah,bay,Tau,ah0test,h0test,bahneverlearntest,bhneverlearntest,bayneverlearntest,dt,INtest,nonlinearity);% dim x numT x numexamples matrix
        normalizederror = 0;
        for ii=1:dimOUT
            yforerror = y(ii,itimeRNNtest(ii,:,:)==1);% 1 x sum(itimeRNNtest(ii,:,:)==1) matrix
            OUTtestforerror = TARGETOUTtest(ii,itimeRNNtest(ii,:,:)==1);% 1 x sum(itimeRNNtest(ii,:,:)==1) matrix
            normalizederror = normalizederror + 100*((OUTtestforerror(:) - yforerror(:))' * (OUTtestforerror(:) - yforerror(:))) / ((OUTtestforerror(:) - mean(OUTtestforerror(:)))'*(OUTtestforerror(:) - mean(OUTtestforerror(:))));% normalized error when using outputs for which itimeRNN = 1
        end
        normalizederror = normalizederror / dimOUT;
        errornormalized_store(i) = normalizederror;
        %---------------------------
        
        msg = sprintf('Epoch %g',epoch); fprintf(repmat('\b',1,numel(msg))); fprintf(msg);
    end% if exist(fullfile(datadir,sprintf('parameters_saveepoch%g.mat',epoch)),'file')==2
end
epochset(error_store==-700) = [];
error_store(error_store==-700) = [];
errornormalized_store(errornormalized_store==-700) = [];
save(fullfile(figuredir,'errornormalized_store.mat'),'errornormalized_store');
save(fullfile(figuredir,'epochset.mat'),'epochset');

handle = figure;% test error as a function of training epoch
fontsize = 15; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',10,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)% only need to set DefaulttextFontSize if using function text
if min(error_store)>=0
    semilogy(epochset,error_store,'k-'); hold on;
    semilogy(epochset([1 end]),error_store([1 end]),'ko')
end
if min(error_store)<0
    plot(epochset,error_store,'k-'); hold on;
    plot(epochset([1 end]),error_store([1 end]),'ko')
end
xlabel('Training epoch')
ylabel('Errormain on test set ')
title({[sprintf('%g test trials, %g time steps in simulation',numtrials,numTtest)];[sprintf('Error epoch %g  = %g, epoch %g = %g',epochset(1),error_store(1),epochset(end),error_store(end))]})
if min(error_store)~=error_store(end)
    ibest = find(error_store == min(error_store));
    plot(epochset([1 ibest end]),error_store([1 ibest end]),'ko')
    title({[sprintf('%g test trials, %g time steps in simulation',numtrials,numTtest)];[sprintf('Error epoch %g  = %g, epoch %g = %g, epoch %g = %g',epochset(1),error_store(1),epochset(ibest),error_store(ibest),epochset(end),error_store(end))]})  
end
axis tight; axis([-Inf Inf min(ylim)-abs(max(ylim)-min(ylim))/100  max(ylim)+abs(max(ylim)-min(ylim))/100])
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
print(handle, '-dpdf', sprintf('%s/error_test_numT%g',figuredir,numTtest))

handle = figure;% percent normalized test error as a function of training epoch
fontsize = 15; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',10,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)% only need to set DefaulttextFontSize if using function text
plot(epochset,errornormalized_store,'k-'); hold on;
plot(epochset([1 end]),errornormalized_store([1 end]),'ko')
xlabel('Training epoch')
ylabel('% normalized test error')
title({[sprintf('%g test trials, %g time steps in simulation',numtrials,numTtest)];[sprintf('Error epoch %g  = %g%%, epoch %g = %g%%',epochset(1),errornormalized_store(1),epochset(end),errornormalized_store(end))]})
if min(errornormalized_store)~=errornormalized_store(end)
    ibest = find(errornormalized_store == min(errornormalized_store));
    plot(epochset([1 ibest end]),errornormalized_store([1 ibest end]),'ko')
    title({[sprintf('%g test trials, %g time steps in simulation',numtrials,numTtest)];[sprintf('Error epoch %g  = %g%%, epoch %g = %g%%, epoch %g = %g%%',epochset(1),errornormalized_store(1),epochset(ibest),errornormalized_store(ibest),epochset(end),errornormalized_store(end))]})  
end
axis tight; axis([-Inf Inf min(ylim)-abs(max(ylim)-min(ylim))/100  max(ylim)+abs(max(ylim)-min(ylim))/100])
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
print(handle, '-dpdf', sprintf('%s/errornormalized_test_numT%g',figuredir,numTtest))

handle = figure;% percent normalized test error as a function of tictoc time
fontsize = 15; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',10,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)% only need to set DefaulttextFontSize if using function text
load(sprintf('%s/tictoctime_store.mat',datadir))
cumsumtictoctime = cumsum(tictoctime_store);
plot(cumsumtictoctime(epochset)/60,errornormalized_store,'k-'); hold on;
plot(cumsumtictoctime(epochset([1 end]))/60,errornormalized_store([1 end]),'ko')
xlabel('Training time (minutes)')
ylabel('% normalized test error')
title({[sprintf('%g test trials, %g time steps in simulation',numtrials,numTtest)];[sprintf('Error epoch %g  = %g%%, epoch %g = %g%%',epochset(1),errornormalized_store(1),epochset(end),errornormalized_store(end))]})
if min(errornormalized_store)~=errornormalized_store(end)
    ibest = find(errornormalized_store == min(errornormalized_store));
    plot(cumsumtictoctime(epochset([1 ibest end]))/60,errornormalized_store([1 ibest end]),'ko')
    title({[sprintf('%g test trials, %g time steps in simulation',numtrials,numTtest)];[sprintf('Error epoch %g  = %g%%, epoch %g = %g%%, epoch %g = %g%%',epochset(1),errornormalized_store(1),epochset(ibest),errornormalized_store(ibest),epochset(end),errornormalized_store(end))]})  
end
axis tight; axis([-Inf Inf min(ylim)-abs(max(ylim)-min(ylim))/100  max(ylim)+abs(max(ylim)-min(ylim))/100])
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
print(handle, '-dpdf', sprintf('%s/errornormalizedVStictoctime_test_numT%g',figuredir,numTtest))


%--------------------------------------------------------------------------
%       load parameters from epoch with smallest errormain on test set
%--------------------------------------------------------------------------
epoch = epochset(error_store==min(error_store));
%epoch = numepoch;
load(fullfile(datadir,sprintf('parameters_saveepoch%g.mat',epoch)));
eval(sprintf('parameterssave = parameters_saveepoch%g;',epoch));
[ah0, h0, Whx, Whh, Wyh, bah, bay, Tau] = unpackall(parameterssave,dimIN,numh,dimOUT,numtrain);
ah0test = ah0(:,1)*ones(1,numtrials);
h0test = h0(:,1)*ones(1,numtrials);
modelgradEtest.ah0 = ah0test;
modelgradEtest.h0 = h0test;
modelgradEtest.Whx = Whx;
modelgradEtest.Whh = Whh;
modelgradEtest.Wyh = Wyh;
modelgradEtest.bah = bah;
modelgradEtest.bay = bay;
modelgradEtest.Tau = Tau;
[~, errormain] = computegradE([],modelgradEtest);
clear modelgradEtest% free up memory
[ah, h, ay, y] = forwardpass(Whx,Whh,Wyh,bah,bay,Tau,ah0test,h0test,bahneverlearntest,bhneverlearntest,bayneverlearntest,dt,INtest,nonlinearity);
%---------------------------
% normalized error, if RNN output is constant for each dimOUT (each dimOUT can be a different constant) then normalizederror = 100%
% yforerror = y(itimeRNNtest==1);
% OUTtestforerror = TARGETOUTtest(itimeRNNtest==1);
% normalizederror = 100*((yforerror(:) - OUTtestforerror(:))' * (yforerror(:) - OUTtestforerror(:))) / ((OUTtestforerror(:) - mean(OUTtestforerror(:)))'*(OUTtestforerror(:) - mean(OUTtestforerror(:))));% normalized error when using outputs for which itimeRNNtest = 1
normalizederror = 0;
errormain_linearoutputnonlinearity = 0;% errormain when output nonlinearity is linear
for i=1:dimOUT
    yforerror = y(i,itimeRNNtest(i,:,:)==1);% 1 x sum(itimeRNNtest(i,:,:)==1) matrix
    OUTtestforerror = TARGETOUTtest(i,itimeRNNtest(i,:,:)==1);% 1 x sum(itimeRNNtest(i,:,:)==1) matrix
    normalizederror = normalizederror + 100*((OUTtestforerror(:) - yforerror(:))' * (OUTtestforerror(:) - yforerror(:))) / ((OUTtestforerror(:) - mean(OUTtestforerror(:)))'*(OUTtestforerror(:) - mean(OUTtestforerror(:))));% normalized error when using outputs for which itimeRNNtest = 1
    errormain_linearoutputnonlinearity = errormain_linearoutputnonlinearity + (OUTtestforerror(:) - yforerror(:))' * (OUTtestforerror(:) - yforerror(:));
end
normalizederror = normalizederror / dimOUT;
errormain_linearoutputnonlinearity = 0.5 * errormain_linearoutputnonlinearity / sum(itimeRNNtest(:)==1);


handle = figure;% eigenvalues of Whh
hold on; fontsize = 15; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',10,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)% only need to set DefaulttextFontSize if using function text
scatter(real(eig(Whh)),imag(eig(Whh)),50,'markerfacecolor','k','markeredgecolor','k');
xlabel('real(eig(Whh))')
ylabel('imag(eig(Whh))')
title(sprintf('Training epoch %g, %g time steps in simulation, errormain = %.5g',epoch,numTtest,errormain)); 
axis tight; axis([-Inf Inf min(ylim)-abs(max(ylim)-min(ylim))/100  max(ylim)+abs(max(ylim)-min(ylim))/100])
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
print(handle, '-dpdf', sprintf('%s/eigenvaluesWhh_epoch%g',figuredir,epoch))

handle = figure;% RNN output on test trials
PLOTINPUT = 1;
Tplot = [1:numTtest];
%for itrial = 1:numtrials
for itrial = [1 numtrials]   
clf; hold on; fontsize = 15; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',10,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)% only need to set DefaulttextFontSize if using function text
A = TARGETOUTtest(:,Tplot,itrial);% dimOUT x numTplot matrix
colors = parula(dimIN);
if PLOTINPUT==1; plot(Tplot,INtest(1,Tplot,itrial),'color',colors(end,:)); plot(Tplot(itimeRNNtest(1,Tplot,itrial)==1),A(1,itimeRNNtest(1,Tplot,itrial)==1),'k-'); plot(Tplot,y(1,Tplot,itrial),'r--'); end% for legend
if PLOTINPUT==0; plot(Tplot(itimeRNNtest(1,Tplot,itrial)==1),A(1,itimeRNNtest(1,Tplot,itrial)==1),'k-'); plot(Tplot,y(1,Tplot,itrial),'r--'); end% for legend
if PLOTINPUT==1; for i=1:dimIN; plot(Tplot,INtest(i,Tplot,itrial),'color',colors(i,:)); end; end
for i=1:dimOUT
    plot(Tplot(itimeRNNtest(i,Tplot,itrial)==1),A(i,itimeRNNtest(i,Tplot,itrial)==1),'k-');% target output
    plot(Tplot,y(i,Tplot,itrial),'r--')% RNN output
end
xlabel('Time steps')
if PLOTINPUT==1; legend('Input','Output: target','Output: RNN','location','best'); end 
if PLOTINPUT==0; legend('Output: target','Output: RNN','location','best'); end 
%title(sprintf('Trial %g, training epoch %g, %g time steps in simulation, errormain = %.5g',itrial,epoch,numTtest,errormain)); 
normalizederror_plotted = 0;
for i=1:dimOUT
    yforerror = y(i,Tplot(itimeRNNtest(i,Tplot,itrial)==1),itrial);% 1 x sum(itimeRNNtest(i,Tplot,itrial)==1) matrix
    OUTtestforerror = TARGETOUTtest(i,Tplot(itimeRNNtest(i,Tplot,itrial)==1),itrial);% 1 x sum(itimeRNNtest(i,Tplot,itrial)==1) matrix
    normalizederror_plotted = normalizederror_plotted + 100*((OUTtestforerror(:) - yforerror(:))' * (OUTtestforerror(:) - yforerror(:))) / ((OUTtestforerror(:) - mean(OUTtestforerror(:)))'*(OUTtestforerror(:) - mean(OUTtestforerror(:))));% normalized error when using outputs for which itimeRNNtest = 1
end
normalizederror_plotted = normalizederror_plotted / dimOUT;
title({[sprintf('Trial %g, training epoch %g, %g time steps in simulation',itrial,epoch,numTtest)];[sprintf('errormain = %.5g, normalized error overall/plotted = %.5g%%/%.5g%%',errormain,normalizederror,normalizederror_plotted)]}); 
axis tight; axis([-Inf Inf min(ylim)-abs(max(ylim)-min(ylim))/100  max(ylim)+abs(max(ylim)-min(ylim))/100])
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
%if numTtest==numel(Tplot); print(handle, '-dpdf', sprintf('%s/testtrial%g_numT%g_bhnoiserandn_0inputphase',figuredir,itrial,numTtest)); end
if numTtest==numel(Tplot); print(handle, '-dpdf', sprintf('%s/testtrial%g_numT%g_epoch%g',figuredir,itrial,numTtest,epoch)); end
if numTtest~=numel(Tplot); print(handle, '-dpdf', sprintf('%s/testtrial%g_numT%g_numTplot%g_epoch%g',figuredir,itrial,numTtest,numel(Tplot),epoch)); end
%pause
end


% RNN output on test trials in degrees
TARGETOUTdegrees = angle_radians * 180/pi;% numTtest x numtrials matrix, target output between 0 and 360 degrees
RNNoutputangle = atan2(squeeze(y(1,:,:)),squeeze(y(2,:,:)))*180/pi;% numTtest x numtrials matrix, angle between -180 and +180 degrees
iswitch = find(RNNoutputangle < 0); RNNoutputangle(iswitch) = 360 + RNNoutputangle(iswitch);% angle between 0 and 360 degrees
handle = figure;% RNN output on test trials in degrees
for itrial = [1 numtrials]  
    clf; hold on; fontsize = 24; set(gcf,'DefaultLineLineWidth',6,'DefaultLineMarkerSize',10,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)% only need to set DefaulttextFontSize if using function text
    idiscontinuityTARGETOUT = find(abs(diff(TARGETOUTdegrees(:,itrial))) > 100);% find and do not plot datapoints where plot jumps from 0 to 360
    idiscontinuityRNN = find(abs(diff(RNNoutputangle(:,itrial))) > 100);% find and do not plot datapoints where plot jumps from 0 to 360
    TARGETOUTdegrees(idiscontinuityTARGETOUT,itrial) = NaN;
    RNNoutputangle(idiscontinuityRNN,itrial) = NaN;
    
    plot(1:numTtest,TARGETOUTdegrees(:,itrial),'k-')
    plot(1:numTtest,RNNoutputangle(:,itrial),'r--')
    xlabel('Timesteps')
    ylabel('Head direction (degrees)')
    legend('Output: target','Output: RNN','location','best');
    title({[sprintf('Trial %g, training epoch %g, %g timesteps in simulation',itrial,epoch,numTtest)];[sprintf('errormain = %.5g, normalized error overall = %.5g%%',errormain,normalizederror)]});
    axis tight; axis([-Inf Inf min(ylim)-abs(max(ylim)-min(ylim))/100  max(ylim)+abs(max(ylim)-min(ylim))/100])
    if max(ylim)>=355; ylim([0 360]); set(gca,'YTick',[0 180 360],'YTickLabel',[0 180 360]); end
    set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
    set(gca,'linewidth',2); set(gca,'TickLength',[0.02 0.025])% default set(gca,'TickLength',[0.01 0.025])
    print(handle, '-dpdf', sprintf('%s/testtrial%g_numT%g_epoch%g_degrees',figuredir,itrial,numTtest,epoch))
    %pause
end% for itrial = [1 numtrials]
    

    

handle = figure;% firing of hidden units
Tplot = [1:numTtest];
for itrial = [1 numtrials]
clf; hold on; fontsize = 15; set(gcf,'DefaultLineLineWidth',1,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize) 
plot(Tplot,h(1,Tplot,itrial),'-')% for legend
plot(Tplot,mean(h(:,Tplot,itrial),1),'k-')% for legend
%plot(Tplot,mean(abs(h(:,Tplot,itrial)),1),'k--')% for legend
plot(Tplot,h(:,Tplot,itrial),'-')
plot(Tplot,mean(h(:,Tplot,itrial),1),'k-','linewidth',2)
%plot(Tplot,mean(abs(h(:,Tplot,itrial)),1),'k--')
xlabel('Time steps')
legend('Firing of hidden units','Mean firing rate','Mean abs firing rate','location','best')
legend('Firing of hidden units','Mean firing rate','location','best')
%legend('Firing of hidden units','location','best')
title(sprintf('Trial %g, training epoch %g, %g time steps, errormain = %.5g',itrial,epoch,numTtest,errormain)); 
axis tight; axis([-Inf Inf -Inf  max(ylim)+abs(max(ylim)-min(ylim))/100])
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
%if numTtest==numel(Tplot); print(handle, '-dpdf', sprintf('%s/testtrial%g_numT%g_h_bhnoiserandn_0inputphase',figuredir,itrial,numTtest)); end
if numTtest==numel(Tplot); print(handle, '-dpdf', sprintf('%s/testtrial%g_numT%g_epoch%g_h',figuredir,itrial,numTtest,epoch)); end
if numTtest~=numel(Tplot); print(handle, '-dpdf', sprintf('%s/testtrial%g_numT%g_numTplot%g_epoch%g_h',figuredir,itrial,numTtest,numel(Tplot),epoch)); end
%pause
end








































