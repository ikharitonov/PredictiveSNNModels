% CJ Cueva, 10.7.2021
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
% bahneverlearn - numh x numT x numexamples matrix, noise added to units before nonlinearity 
% bhneverlearn  - numh x numT x numexamples matrix, noise added to units after nonlinearity
% bayneverlearn - dimOUT x numT x numexamples matrix, noise added to outputs
close all; 
clear all;
% Change datadir to the folder location on your computer. After
% this change the program should run and generate the figures I sent you.
datadir = '/Users/christophercueva/Desktop/forIakovKharitonov';
figuredir = datadir;% save figures to figuredir
addpath(genpath(datadir))% add folder and all of its subfolders to the top of the search path
cd(datadir)


%--------------------------------------------------------------------------
%                          RNN parameters
%--------------------------------------------------------------------------
dimIN = 3;% number of inputs, input 1 is angular velocity (angle to integrate), input 2 is sin(angle0), input 3 is cos(angle0)
numh = 100;% number of hidden units
dimOUT = 2;% number of outputs, output 1 is sin(integrated-angle), output 2 is cos(integrated-angle)
dt = 1; Tau = 10*ones(numh,1);% neural time-constant 
nonlinearity{1} = 'retanh';% nonlinearity{1} is the nonlinearity of the hidden units 
nonlinearity{2} = 'linear';% nonlinearity{2} is the nonlinearity of the output units
numtrain = 500;% number of training trials

%--------------------------------------------------------------------------
%               Generate inputs to RNN and target outputs
%--------------------------------------------------------------------------
angle0duration = 10;% angle0 input (sin(angle0) and cos(angle0)) is nonzero for angle0duration timesteps at beginning of trial
ANGULARVELOCITY.angularmomentum = 0.8;% angularvelocity(t) = sd*randn + angularmomentum * angularvelocity(t-1), angularmomentum and sd are not used when using ANGULARVELOCITY.discreteangularvelocitydegrees
ANGULARVELOCITY.sd = .03;% at each timestep the new angularvelocity is a gaussian random variable with mean 0 and standard deviation sd, angularmomentum and sd are not used when using ANGULARVELOCITY.discreteangularvelocitydegrees
ANGULARVELOCITY.angularvelocitymindegrees = -Inf;% minimum angular input/angular-velocity on a single timestep (degrees)
ANGULARVELOCITY.angularvelocitymaxdegrees = Inf;% maximum angular input/angular-velocity on a single timestep (degrees)
noiseamplitude_input = 0;
BOUNDARY.periodic = 1;% if 1 periodic boundary conditions

numtrials = 2000;
numTtest = 1000;
randseedtest = 99999999;
[INtest, TARGETOUTtest, itimeRNNtest, angle_radians] = generateINandTARGETOUT(dimIN,dimOUT,numTtest,numtrials,randseedtest,noiseamplitude_input,angle0duration,ANGULARVELOCITY,BOUNDARY);
% IN:         dimIN x numT x numtrials matrix
% TARGETOUT:  dimOUT x numT x numtrials matrix
% itimeRNN:   dimOUT x numT x numtrials matrix, elements 0(time-point does not contribute to first term in cost function), 1(time-point contributes to first term in cost function)
%bahneverlearntest = zeros(numh,numTtest,numtrials);
%bhneverlearntest = 0.1*randn(numh,numTtest,numtrials);% add noise to units
%bayneverlearntest = zeros(dimOUT,numTtest,numtrials);   
bahneverlearntest = 0; bhneverlearntest = 0; bayneverlearntest = 0;% set to 0 instead of zeros(...,numT,numexamples) to save memory and run faster, code is written so this only works if all never-learned biases are 0


%--------------------------------------------------------------------------
%       load RNN parameters and generate unit activity
%--------------------------------------------------------------------------
epoch = 415;
load(fullfile(datadir,sprintf('parameters_saveepoch%g.mat',epoch)));
eval(sprintf('parameterssave = parameters_saveepoch%g;',epoch));
[ah0, h0, Whx, Whh, Wyh, bah, bay, Tau] = unpackall(parameterssave,dimIN,numh,dimOUT,numtrain);
ah0test = ah0(:,1)*ones(1,numtrials);
h0test = h0(:,1)*ones(1,numtrials);
[ah, h, ay, y] = forwardpass(Whx,Whh,Wyh,bah,bay,Tau,ah0test,h0test,bahneverlearntest,bhneverlearntest,bayneverlearntest,dt,INtest,nonlinearity);
% h is the unit activity after the nonlinearity
% h has size number-of-units x number-of-timesteps x number-of-trials
%---------------------------
% normalized error, if RNN output is constant for each dimOUT (each dimOUT can be a different constant) then normalizederror = 100%
% yforerror = y(itimeRNNtest==1);
% OUTtestforerror = TARGETOUTtest(itimeRNNtest==1);
% normalizederror = 100*((yforerror(:) - OUTtestforerror(:))' * (yforerror(:) - OUTtestforerror(:))) / ((OUTtestforerror(:) - mean(OUTtestforerror(:)))'*(OUTtestforerror(:) - mean(OUTtestforerror(:))));% normalized error when using outputs for which itimeRNNtest = 1
normalizederror = 0;
for i=1:dimOUT
    yforerror = y(i,itimeRNNtest(i,:,:)==1);% 1 x sum(itimeRNNtest(i,:,:)==1) matrix
    OUTtestforerror = TARGETOUTtest(i,itimeRNNtest(i,:,:)==1);% 1 x sum(itimeRNNtest(i,:,:)==1) matrix
    normalizederror = normalizederror + 100*((OUTtestforerror(:) - yforerror(:))' * (OUTtestforerror(:) - yforerror(:))) / ((OUTtestforerror(:) - mean(OUTtestforerror(:)))'*(OUTtestforerror(:) - mean(OUTtestforerror(:))));% normalized error when using outputs for which itimeRNNtest = 1
end
normalizederror = normalizederror / dimOUT;
 


handle = figure;% eigenvalues of recurrent weight matrix Whh
hold on; fontsize = 15; set(gcf,'DefaultLineLineWidth',2,'DefaultLineMarkerSize',10,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)% only need to set DefaulttextFontSize if using function text
scatter(real(eig(Whh)),imag(eig(Whh)),50,'markerfacecolor','k','markeredgecolor','k');
xlabel('real(eig(Whh))')
ylabel('imag(eig(Whh))')
title(sprintf('Training epoch %g, %g time steps in simulation, normalized error = %.5g',epoch,numTtest,normalizederror)); 
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
normalizederror_plotted = 0;
for i=1:dimOUT
    yforerror = y(i,Tplot(itimeRNNtest(i,Tplot,itrial)==1),itrial);% 1 x sum(itimeRNNtest(i,Tplot,itrial)==1) matrix
    OUTtestforerror = TARGETOUTtest(i,Tplot(itimeRNNtest(i,Tplot,itrial)==1),itrial);% 1 x sum(itimeRNNtest(i,Tplot,itrial)==1) matrix
    normalizederror_plotted = normalizederror_plotted + 100*((OUTtestforerror(:) - yforerror(:))' * (OUTtestforerror(:) - yforerror(:))) / ((OUTtestforerror(:) - mean(OUTtestforerror(:)))'*(OUTtestforerror(:) - mean(OUTtestforerror(:))));% normalized error when using outputs for which itimeRNNtest = 1
end
normalizederror_plotted = normalizederror_plotted / dimOUT;
title({[sprintf('Trial %g, training epoch %g, %g time steps in simulation',itrial,epoch,numTtest)];[sprintf('normalized error overall/plotted = %.5g%%/%.5g%%',normalizederror,normalizederror_plotted)]}); 
axis tight; axis([-Inf Inf min(ylim)-abs(max(ylim)-min(ylim))/100  max(ylim)+abs(max(ylim)-min(ylim))/100])
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
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
    clf; hold on; fontsize = 20; set(gcf,'DefaultLineLineWidth',6,'DefaultLineMarkerSize',10,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)% only need to set DefaulttextFontSize if using function text
    idiscontinuityTARGETOUT = find(abs(diff(TARGETOUTdegrees(:,itrial))) > 100);% find and do not plot datapoints where plot jumps from 0 to 360
    idiscontinuityRNN = find(abs(diff(RNNoutputangle(:,itrial))) > 100);% find and do not plot datapoints where plot jumps from 0 to 360
    TARGETOUTdegrees(idiscontinuityTARGETOUT,itrial) = NaN;
    RNNoutputangle(idiscontinuityRNN,itrial) = NaN;
    
    plot(1:numTtest,TARGETOUTdegrees(:,itrial),'k-')
    plot(1:numTtest,RNNoutputangle(:,itrial),'r--')
    xlabel('Timesteps')
    ylabel('Head direction (degrees)')
    legend('Output: target','Output: RNN','location','best');
    title({[sprintf('Trial %g, training epoch %g, %g timesteps in simulation',itrial,epoch,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)]});
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
title(sprintf('Trial %g, training epoch %g, %g time steps, normalized error = %.5g',itrial,epoch,numTtest,normalizederror)); 
axis tight; axis([-Inf Inf -Inf  max(ylim)+abs(max(ylim)-min(ylim))/100])
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
%if numTtest==numel(Tplot); print(handle, '-dpdf', sprintf('%s/testtrial%g_numT%g_h_bhnoiserandn_0inputphase',figuredir,itrial,numTtest)); end
if numTtest==numel(Tplot); print(handle, '-dpdf', sprintf('%s/testtrial%g_numT%g_epoch%g_h',figuredir,itrial,numTtest,epoch)); end
if numTtest~=numel(Tplot); print(handle, '-dpdf', sprintf('%s/testtrial%g_numT%g_numTplot%g_epoch%g_h',figuredir,itrial,numTtest,numel(Tplot),epoch)); end
%pause
end


%%
%--------------------------------------------------------------------------
%                   input/movement statistics
%   distribution of angular input and integrated angular input
%--------------------------------------------------------------------------
eps = 10^-4;% add epsilon to largest element of anglegrid so when angle equals max(anglegrid) then iangle is not empty 
dangle1 = 1; angle1grid = [0:dangle1:(360-dangle1) 360+eps];% 1 x 361 matrix, discretize the integrated angle
if BOUNDARY.periodic == 0; dangle1 = 1; angle1grid = [BOUNDARY.minangle:dangle1:(BOUNDARY.maxangle-dangle1) BOUNDARY.maxangle+eps]; end% integrated angle cannot go outside BOUNDARY.minangle and BOUNDARY.maxangle
minangularinput = min(min(INtest(1,:,:))) * 180/pi; maxangularinput = max(max(INtest(1,:,:))) * 180/pi; dangle2 = (maxangularinput - minangularinput)/100; angle2grid = [minangularinput:dangle2:(maxangularinput-dangle2) maxangularinput+eps];% 1 x something matrix, discretize the angular input

numvisits = zeros(numel(angle2grid)-1,numel(angle1grid)-1);% number of times agent visits bin
for itrial=1:numtrials
    for t=1:numTtest
        angle1 = angle_radians(t,itrial)*180/pi;% stored angle in degrees, between 0 and 360
        angle2 = INtest(1,t,itrial)*180/pi;% angular input in degrees
        iangle1 = find(angle1 < angle1grid,1,'first') - 1;% if xposition < min(angle1grid) iangle1 = 0, if xposition > max(angle1grid) iangle1 is an empty matrix
        iangle2 = find(angle2 < angle2grid,1,'first') - 1;% if yposition < min(angle2grid) iangle2 = 0, if yposition > max(angle2grid) iangle2 is an empty matrix
        numvisits(iangle2,iangle1) = numvisits(iangle2,iangle1) + 1;
    end
end
if sum(numvisits(:)) ~= numtrials*numTtest; error('missing visits'); end
    
numvisits_remove0angularinput = numvisits;
i0angularinput = find(0 < angle2grid, 1,'first') - 1;% index where 0 angular input is stored
numvisits_remove0angularinput(i0angularinput,:) = NaN;

for iteration=1:2
    handle = figure;% input/movement distribution
    clf; hold on; fontsize = 15; set(gcf,'DefaultLineLineWidth',1,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
    if iteration==1; A = numvisits; end% A(1,1) is in the top left corner of the image, if not set(gca,'YDir','reverse')
    if iteration==2; A = numvisits_remove0angularinput; end% A(1,1) is in the top left corner of the image, if not set(gca,'YDir','reverse')
    handleimage = imagesc(angle1grid(1:end-1)+dangle1/2,angle2grid(1:end-1)+dangle2/2,A);
    %set(gca,'YDir','normal')% flip so point of triangle is at top
    set(handleimage,'alphadata',~isnan(A))% if numvisits is 0 firingrate is NaN
    colormap parula;
    handle_colorbar = colorbar('FontSize',fontsize); set(get(handle_colorbar,'ylabel'),'string','Number of visits','FontSize',fontsize);% hsv, jet

    xlabel(sprintf('Integrated angular input (%.3g to %.3g degrees)',min(angle1grid),max(angle1grid)))
    ylabel((sprintf('Angular input (%.3g to %.3g degrees)',min(angle2grid),max(angle2grid))))
    title({['Distribution of angular input and integrated angular input'];[sprintf('Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)]})
    axis tight;
    set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
    if iteration==1; print(handle, '-dpdf', sprintf('%s/inputandmovementstatistics_epoch%g_numT%g',figuredir,epoch,numTtest)); end
    if iteration==2; print(handle, '-dpdf', sprintf('%s/inputandmovementstatistics_epoch%g_numT%g_remove0angularinput',figuredir,epoch,numTtest)); end
end

handle = figure;% input distribution
clf; hold on; fontsize = 22; set(gcf,'DefaultLineLineWidth',4,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
plot(angle2grid(1:end-1)+dangle2/2,sum(numvisits,2),'k-')    
xlabel((sprintf('Angular input (%.3g to %.3g degrees)',min(angle2grid),max(angle2grid))))
ylabel('Frequency')   
title({['Distribution of angular inputs to RNN'];[sprintf('Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)]})
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
axis tight; axis([-Inf Inf 0 max(ylim)+abs(max(ylim)-min(ylim))/100])
set(gca,'linewidth',2)
print(handle, '-dpdf', sprintf('%s/inputstatistics_epoch%g_numT%g',figuredir,epoch,numTtest))
      

%%
%--------------------------------------------------------------------------
%  plot the firing rates of each unit as a function of the stored angle (integrated input)
%  integrated angle is computed in two ways: 1) based on integrated inputs (ground truth) and 2) based on RNN outputs (internal estimate of heading direction)
%--------------------------------------------------------------------------
close all
dangle = 1; anglegrid = [0:dangle:360];% 1 x 361 matrix, discretize the angle
if BOUNDARY.periodic == 0; dangle = 1; anglegrid = [BOUNDARY.minangle:dangle:BOUNDARY.maxangle]; end% integrated angle cannot go outside BOUNDARY.minangle and BOUNDARY.maxangle

firingrate_store = -700*ones(numh,numel(anglegrid)-1);
firingrate_posinput_store = -700*ones(numh,numel(anglegrid)-1);% firing rate when input angle is positive
firingrate_neginput_store = -700*ones(numh,numel(anglegrid)-1);% firing rate when input angle is negative
firingrate_integratedanglefromRNNoutput_store = -700*ones(numh,numel(anglegrid)-1);
maxminusmin_integratedangularinput = -700*ones(numh,1);% strength of tuning to integrated angular input
maxminusmin_integratedangularinputfromRNNoutput = -700*ones(numh,1);% strength of tuning to integrated angular input
anglepreferred_integratedangularinput = -700*ones(numh,1);% integrated angle (in degrees) that yields maximum firing,  NOT ANYMORE!technically maximum absolute deviation from the mean (include this to better capture units that have high baseline firing and dip down for preferred angle)
anglepreferred_integratedangularinput_posinput = -700*ones(numh,1);% integrated angle (in degrees) that yields maximum firing, NOT ANYMORE!technically maximum absolute deviation from the mean (include this to better capture units that have high baseline firing and dip down for preferred angle)
anglepreferred_integratedangularinput_neginput = -700*ones(numh,1);% integrated angle (in degrees) that yields maximum firing,  NOT ANYMORE!technically maximum absolute deviation from the mean (include this to better capture units that have high baseline firing and dip down for preferred angle)
anglepreferred_integratedangularinputfromRNNoutput = -700*ones(numh,1);% integrated angle (in degrees) that yields maximum firing,  NOT ANYMORE!technically maximum absolute deviation from the mean (include this to better capture units that have high baseline firing and dip down for preferred angle)

for iunit=1:numh    
    firingrate = zeros(1,numel(anglegrid)-1);
    firingrate_posinput = zeros(1,numel(anglegrid)-1);
    firingrate_neginput = zeros(1,numel(anglegrid)-1);
    firingrate_integratedanglefromRNNoutput = zeros(1,numel(anglegrid)-1);
    numvisits = zeros(1,numel(anglegrid)-1);% number of times RNN stores a specific angular input
    numvisits_posinput = zeros(1,numel(anglegrid)-1);% number of times RNN stores a specific angular input
    numvisits_neginput = zeros(1,numel(anglegrid)-1);% number of times RNN stores a specific angular input
    numvisits_integratedanglefromRNNoutput = zeros(1,numel(anglegrid)-1);% number of times RNN stores a specific angular input
    for itrial=1:numtrials
        RNNoutputangle = -700*ones(1,numTtest);
        RNNoutputangle(:) = atan2(y(1,:,itrial),y(2,:,itrial))*180/pi;% angle between -180 and +180 degrees
        iswitch = find(RNNoutputangle < 0); RNNoutputangle(iswitch) = 360 + RNNoutputangle(iswitch);% angle between 0 and 360 degrees
        for t=1:numTtest
            angle = angle_radians(t,itrial)*180/pi;% stored angle in degrees
            iposition = find(angle < anglegrid ,1,'first') - 1;
            numvisits(iposition) = numvisits(iposition) + 1;
            firingrate(iposition) = firingrate(iposition) + h(iunit,t,itrial);
            
            inputangle = INtest(1,t,itrial);% angular input in radians
            if inputangle>0
                numvisits_posinput(iposition) = numvisits_posinput(iposition) + 1;
                firingrate_posinput(iposition) = firingrate_posinput(iposition) + h(iunit,t,itrial);
            end
            if inputangle<0
                numvisits_neginput(iposition) = numvisits_neginput(iposition) + 1;
                firingrate_neginput(iposition) = firingrate_neginput(iposition) + h(iunit,t,itrial);
            end
            
            angle = RNNoutputangle(t);% stored angle in degrees
            iposition = find(angle < anglegrid ,1,'first') - 1;
            numvisits_integratedanglefromRNNoutput(iposition) = numvisits_integratedanglefromRNNoutput(iposition) + 1;
            firingrate_integratedanglefromRNNoutput(iposition) = firingrate_integratedanglefromRNNoutput(iposition) + h(iunit,t,itrial);
        end
    end
    if (sum(numvisits(:)) ~= numtrials*numTtest); error('missing visits'); end
    if (sum(numvisits_integratedanglefromRNNoutput(:)) ~= numtrials*numTtest); error('missing visits'); end
    firingrate = firingrate ./ numvisits;
    firingrate_posinput = firingrate_posinput ./ numvisits_posinput;
    firingrate_neginput = firingrate_neginput ./ numvisits_neginput;
    firingrate_integratedanglefromRNNoutput = firingrate_integratedanglefromRNNoutput ./ numvisits_integratedanglefromRNNoutput;
    firingrate_store(iunit,:) = firingrate;
    firingrate_posinput_store(iunit,:) = firingrate_posinput;
    firingrate_neginput_store(iunit,:) = firingrate_neginput;
    firingrate_integratedanglefromRNNoutput_store(iunit,:) = firingrate_integratedanglefromRNNoutput;
    maxminusmin_integratedangularinput(iunit) = max(firingrate) - min(firingrate);% strength of tuning to integrated angular input
    maxminusmin_integratedangularinputfromRNNoutput(iunit) = max(firingrate_integratedanglefromRNNoutput) - min(firingrate_integratedanglefromRNNoutput);% strength of tuning to integrated angular input
    
end% for iunit=1:numh

% for iteration 4 sort units by compass, shiftpos, shiftneg, weakly tuned, remove
if exist(fullfile(datadir,'iunits_compass_sort.mat'),'file')==2; load(fullfile(datadir,'iunits_compass_sort.mat')); end
if exist(fullfile(datadir,'iunits_shiftpos_sort.mat'),'file')==2; load(fullfile(datadir,'iunits_shiftpos_sort.mat')); end
if exist(fullfile(datadir,'iunits_shiftneg_sort.mat'),'file')==2; load(fullfile(datadir,'iunits_shiftneg_sort.mat')); end
if exist(fullfile(datadir,'iunits_weaklytuned_sort.mat'),'file')==2; load(fullfile(datadir,'iunits_weaklytuned_sort.mat')); end
if exist(fullfile(datadir,'iunits_remove.mat'),'file')==2; load(fullfile(datadir,'iunits_remove.mat')); end
iteration_set = [1:3];
if exist('iunits_compass_sort','var') && exist('iunits_shiftpos_sort','var') && exist('iunits_shiftneg_sort','var') && exist('iunits_weaklytuned_sort','var') && exist('iunits_remove','var')
    desiredorder = [iunits_compass_sort; iunits_shiftpos_sort; iunits_shiftneg_sort; iunits_weaklytuned_sort; iunits_remove];
    if ~isequal(sort(desiredorder),[1:numh]'); error('missing units, line 1500'); end
    iteration_set = [1:4];
end
for iteration=iteration_set
    handle = figure;% firing of all hidden units as a function of integrated input (true value or RNN output)
    clf; hold on; fontsize = 12; set(gcf,'DefaultLineLineWidth',4,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
    for iunit=1:numh
        if iteration==1; firingrate = firingrate_store(iunit,:); end
        if iteration==2; firingrate = firingrate_integratedanglefromRNNoutput_store(iunit,:); end
        if iteration==3; firingrate_posinput = firingrate_posinput_store(iunit,:); firingrate_neginput = firingrate_neginput_store(iunit,:); end
        if iteration==4; firingrate = firingrate_store(iunit,:); end
        
        % calculate preferred integrated angle
        if iteration==1 || iteration==2
            [maxfiringrate, indexmaxfiringrate] = max( firingrate );% integrated angle that yields maximum firing
            indexpreferredangle = indexmaxfiringrate;
            if iteration==1; anglepreferred_integratedangularinput(iunit) = anglegrid(indexpreferredangle) + dangle/2; end% integrated angle that yields maximum firing, technically maximum absolute deviation from the mean (include this to better capture units that have high baseline firing and dip down for preferred angle)
            if iteration==2; anglepreferred_integratedangularinputfromRNNoutput(iunit) = anglegrid(indexpreferredangle) + dangle/2; end% integrated angle that yields maximum firing, technically maximum absolute deviation from the mean (include this to better capture units that have high baseline firing and dip down for preferred angle)
        end
        if iteration==3% copy code above twice, once for posinput and once for neginput
            firingrate = firingrate_posinput;
            [maxfiringrate, indexmaxfiringrate] = max( firingrate );% integrated angle that yields maximum firing
            indexpreferredangle = indexmaxfiringrate;
            anglepreferred_integratedangularinput_posinput(iunit) = anglegrid(indexpreferredangle) + dangle/2;% integrated angle that yields maximum firing, technically maximum absolute deviation from the mean (include this to better capture units that have high baseline firing and dip down for preferred angle)
            indexpreferredangle_posinput = indexpreferredangle;
            firingrate = firingrate_neginput;
            [maxfiringrate, indexmaxfiringrate] = max( firingrate );% integrated angle that yields maximum firing
            indexpreferredangle = indexmaxfiringrate;
            anglepreferred_integratedangularinput_neginput(iunit) = anglegrid(indexpreferredangle) + dangle/2;% integrated angle that yields maximum firing, technically maximum absolute deviation from the mean (include this to better capture units that have high baseline firing and dip down for preferred angle)
            indexpreferredangle_neginput = indexpreferredangle;
        end
        
        numrowsinfigure = ceil(sqrt(numh));
        numcolumnsinfigure = ceil(numh/numrowsinfigure);
        positionofunit = iunit;% unit iunit is plotted at position positionofunit
        if iteration==4; positionofunit = find(desiredorder == iunit); end% unit iunit is plotted at position positionofunit
        subplot(numrowsinfigure,numcolumnsinfigure,positionofunit)
        if iteration==1 || iteration==2 || iteration==4; plot(anglegrid(1:end-1)+dangle/2,firingrate,'k-'); hold on; end
        if iteration==1; plot(anglepreferred_integratedangularinput(iunit),firingrate(indexpreferredangle),'r.'); end
        if iteration==2; plot(anglepreferred_integratedangularinputfromRNNoutput(iunit),firingrate(indexpreferredangle),'r.'); end
        if iteration==3 
            plot(anglegrid(1:end-1)+dangle/2,firingrate_posinput,'b-','linewidth',2); hold on;% CCW 
            plot(anglegrid(1:end-1)+dangle/2,firingrate_neginput,'g-','linewidth',2)% CW
            relativeangle = mod(anglepreferred_integratedangularinput_posinput(iunit) - anglepreferred_integratedangularinput_neginput(iunit),360);% a number between 0 and 359
            if relativeangle > 180; relativeangle = relativeangle - 360; end% a number between -179 and 180
            title(sprintf('%g%s',relativeangle,char(176)),'fontsize',2,'fontweight','normal');
        end
        
      
        xlim([min(anglegrid) max(anglegrid)])
        if isequal(nonlinearity{1},'tanh'); ylim([-1 1]); end
        if isequal(nonlinearity{1},'retanh'); ylim([0 1]); end
        if isequal(nonlinearity{1},'logistic'); ylim([0 1]); end
        set(gca,'xtick',[],'ytick',[]);
        drawnow
        %pause
        %if rem(iunit,round(numh/10))==1; handlewaitbar = waitbar(iunit/numh); end% waitbar is very slow, redraw bar a maximum of ~10 times
    end% for iunit=1:numh
    if iteration==1 || iteration==3 || iteration==4; [ax1,h1]=suplabel(sprintf('Integrated angular input (%.3g to %.3g degrees)',min(anglegrid),max(anglegrid)),'x'); end
    if iteration==2; [ax1,h1]=suplabel(sprintf('Integrated angular input from RNN output (%.3g to %.3g degrees)',min(anglegrid),max(anglegrid)),'x'); end
    if isequal(nonlinearity{1},'tanh'); [ax2,h2]=suplabel('Activity of unit (-1 to 1)','y'); end
    if isequal(nonlinearity{1},'retanh'); [ax2,h2]=suplabel('Activity of unit (0 to 1)','y'); end
    if isequal(nonlinearity{1},'logistic'); [ax2,h2]=suplabel('Activity of unit (0 to 1)','y'); end
    [a3, h3] = suplabel({[sprintf('Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)]},'t');
    if iteration==3; legend('pos input','neg input','autoupdate','off','location','best'); end% call legend here so suplabel does not raise an error
    set(h1,'fontsize',fontsize); set(h2,'fontsize',fontsize); set(h3,'fontsize',fontsize,'fontweight','normal')
    drawnow
    if iteration==1; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSintegratedangularinput_epoch%g_numT%g',figuredir,epoch,numTtest)); end
    if iteration==2; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSintegratedangularinput_epoch%g_numT%g_integratedanglefromRNNoutput',figuredir,epoch,numTtest)); end 
    if iteration==3; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSintegratedangularinput_epoch%g_numT%g_groupposneginput',figuredir,epoch,numTtest)); end 
    if iteration==4; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSintegratedangularinput_epoch%g_numT%g_sortcompassshiftposshiftneg',figuredir,epoch,numTtest)); end 

end% for iteration=1:4
save(fullfile(figuredir,'anglepreferred_integratedangularinput.mat'),'anglepreferred_integratedangularinput')
save(fullfile(figuredir,'anglepreferred_integratedangularinput_posinput.mat'),'anglepreferred_integratedangularinput_posinput')
save(fullfile(figuredir,'anglepreferred_integratedangularinput_neginput.mat'),'anglepreferred_integratedangularinput_neginput')
save(fullfile(figuredir,'anglepreferred_integratedangularinputfromRNNoutput.mat'),'anglepreferred_integratedangularinputfromRNNoutput')
save(fullfile(figuredir,'maxminusmin_integratedangularinput.mat'),'maxminusmin_integratedangularinput')
save(fullfile(figuredir,'maxminusmin_integratedangularinputfromRNNoutput.mat'),'maxminusmin_integratedangularinputfromRNNoutput')
max_integratedangularinput_posinput = max(firingrate_posinput_store,[],2);% numh x 1 matrix, maximum of tuning curve for integrated-angle (in degrees) when angular velocity input is positive (CCW)
max_integratedangularinput_neginput = max(firingrate_neginput_store,[],2);% numh x 1 matrix, maximum of tuning curve for integrated-angle (in degrees) when angular velocity input is negative (CW)
save(fullfile(figuredir,'max_integratedangularinput_posinput.mat'),'max_integratedangularinput_posinput')
save(fullfile(figuredir,'max_integratedangularinput_neginput.mat'),'max_integratedangularinput_neginput')





%%
%--------------------------------------------------------------------------
%  plot the firing rates of each unit as a function of the angular input 
%--------------------------------------------------------------------------
% only compute tuning when angular input is within two standard deviations of mean
A = INtest(1,:,:)*180/pi;% angular input in degrees
A(A==0) = [];% remove the many 0 elements 
stdangularinput = std(A(:));% standard deviation in degrees

eps = 10^-4;% add epsilon to largest element of anglegrid so when angle equals max(anglegrid) then iangle is not empty
minangularinput = min(min(INtest(1,:,:))) * 180/pi; maxangularinput = max(max(INtest(1,:,:))) * 180/pi; dangle = (maxangularinput - minangularinput)/100; anglegrid = [minangularinput:dangle:(maxangularinput-dangle) maxangularinput+eps];% 1 x something matrix, discretize the angular input
minfortuning = max(-3*stdangularinput,minangularinput); maxfortuning = min(3*stdangularinput,maxangularinput);% if 3*stdangularinput is outside range of input then use min/max of input
ikeep_anglegrid = ((minfortuning <= anglegrid) & (anglegrid <= maxfortuning)); 
if unique(ikeep_anglegrid)==1; ikeep_anglegrid = true(1,numel(anglegrid)-1); end% if ikeep_anglegrid is all 1s make sure ikeep_anglegrid has the same size as firingrate
maxminusmin_angularinput = -700*ones(numh,1);% strength of tuning to angular input
maxminusmin_angularinput_restricedangularrange = -700*ones(numh,1);% strength of tuning to angular input, when angular input is between minfortuning and maxfortuning
slope_firingratevsangularinput = -700*ones(numh,1);% slope when a linear fit is made to the firing rate versus angular input curve
handle = figure;% firing of all hidden units 
clf; hold on; fontsize = 12; set(gcf,'DefaultLineLineWidth',4,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)  
for iunit=1:numh     
    firingrate = zeros(1,numel(anglegrid)-1);
    numvisits = zeros(1,numel(anglegrid)-1);% number of times RNN receives a specific angular input
    for itrial=1:numtrials
        for t=1:numTtest
            angle = INtest(1,t,itrial)*180/pi;% angular input in degrees
            iposition = find(angle < anglegrid ,1,'first') - 1;
            numvisits(iposition) = numvisits(iposition) + 1;
            firingrate(iposition) = firingrate(iposition) + h(iunit,t,itrial);
            
        end
    end
    if (sum(numvisits(:)) ~= numtrials*numTtest); error('missing visits'); end
    firingrate = firingrate ./ numvisits;
    maxminusmin_angularinput(iunit) = max(firingrate) - min(firingrate);% strength of tuning to angular input
    maxminusmin_angularinput_restricedangularrange(iunit) = max(firingrate(ikeep_anglegrid)) - min(firingrate(ikeep_anglegrid));% strength of tuning to angular input, when angular input is between minfortuning and maxfortuning
    %display(sprintf('strength of tuning to angular input (all/restriced range) = %.2g/%.2g',maxminusmin_angularinput(iunit),maxminusmin_angularinput_restricedangularrange(iunit)))% 3 significant digits
    
    % Determine the slope with a linear fit, don't fit portion of curve near 0, if entire curve is near 0 then fit this
    % Fit the target output y_=firingrate by linearly weighting regressors x_=anglegrid(1:end-1)+dangle/2
    % if numvisits is 0 then firingrate is NaN, remove NaN before linear regression
    x_ = [anglegrid(1:end-1)+dangle/2]';% numtrainingexamples x 1 matrix
    y_ = firingrate';% numtrainingexamples x 1 matrix, noisy data
    % remove firing rate when angular input is 0, there is a weird bump 
    i0 = find(0 < anglegrid,1,'first') - 1;% this index is only good if y_ = firingrate, i.e. if y_ hasn't been changed by removing some elements
    x_(i0) = [];% remove 0
    y_(i0) = [];% remove 0
    inan = find(isnan(y_));
    x_(inan) = [];% remove NaN
    y_(inan) = [];% remove NaN
    iclose1 = find(y_>=0.95);
    iclose0 = find(y_<0.05);
    %if (numel(y_)-numel(iclose0)-numel(iclose1)) > numel(y_)/4; iremove = [iclose0; iclose1]; x_(iremove) = []; y_(iremove) = []; end% if more than one quarter of curve is away from 0 or 1 then fit this portion after removing curve near 0 and 1
    if (numel(y_)-numel(iclose0)) > numel(y_)/4; iremove = iclose0; x_(iremove) = []; y_(iremove) = []; end% if more than one quarter of curve is away from 0 then fit this portion after removing curve near 0
    numtrainingexamples = length(x_);
    X = [ones(numtrainingexamples,1) x_];% numtrainingexamples x 2 matrix
    %w = (X'*X)\X'*y;% inv(X'*X)*X'*y, w that minimizes the squared error between y and X*w 
    w = pinv(X)*y_;% w that minimizes the squared error between y and X*w and has the smallest norm w'*w
    firingrate_predictions_xdata = x_;
    firingrate_predictions_ydata = X*w;% numtrainingexamples x 1 matrix
    slope_firingratevsangularinput(iunit) = w(2);


    numrowsinfigure = ceil(sqrt(numh));
    numcolumnsinfigure = ceil(numh/numrowsinfigure);
    subplot(numrowsinfigure,numcolumnsinfigure,iunit); hold on;
    xplot = anglegrid(1:end-1)+dangle/2; yplot = firingrate;
    inan = find(isnan(yplot)); xplot(inan) = []; yplot(inan) = [];% remove NaN
    plot(xplot,yplot,'k-')
    plot(firingrate_predictions_xdata,firingrate_predictions_ydata,'r-','linewidth',1)
    %axis off
    xlim([min(anglegrid) max(anglegrid)])
    if isequal(nonlinearity{1},'tanh'); ylim([-1 1]); end
    if isequal(nonlinearity{1},'retanh'); ylim([0 1]); end
    if isequal(nonlinearity{1},'logistic'); ylim([0 1]); end
    plot(3*stdangularinput*ones(1,100),linspace(min(ylim),max(ylim),100),'k--','linewidth',1)% 3*std is the cutoff for computing tuning to angular input
    plot(-3*stdangularinput*ones(1,100),linspace(min(ylim),max(ylim),100),'k--','linewidth',1)% 3*std is the cutoff for computing tuning to angular input
    set(gca,'xtick',[],'ytick',[]);
    title(sprintf('slope = %.2g, %g%s',slope_firingratevsangularinput(iunit),anglepreferred_integratedangularinput(iunit),char(176)),'fontsize',2,'fontweight','normal');% ax = gca; ax.FontSize = 1;
    drawnow
    %pause
    %if rem(iunit,round(numh/10))==1; handlewaitbar = waitbar(iunit/numh); end% waitbar is very slow, redraw bar a maximum of ~10 times
end% for iunit=1:numh
%if exist('handlewaitbar','var'); close(handlewaitbar); clear handlewaitbar; end  
[ax1,h1]=suplabel(sprintf('Angular input to RNN (%.3g to %.3g degrees)',min(anglegrid),max(anglegrid)),'x');
if isequal(nonlinearity{1},'tanh'); [ax2,h2]=suplabel('Activity of unit (-1 to 1)','y'); end
if isequal(nonlinearity{1},'retanh'); [ax2,h2]=suplabel('Activity of unit (0 to 1)','y'); end
if isequal(nonlinearity{1},'logistic'); [ax2,h2]=suplabel('Activity of unit (0 to 1)','y'); end
[a3, h3] = suplabel({[sprintf('Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)]},'t');
set(h1,'fontsize',fontsize); set(h2,'fontsize',fontsize); set(h3,'fontsize',fontsize,'fontweight','normal')
%set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinput_epoch%g_numT%g',figuredir,epoch,numTtest))




%--------------------------------------------------------------------------
%  plot the firing rates of each unit as a function of the angular input 
%      at the preferred integrated angle for each unit 
%--------------------------------------------------------------------------
% only compute tuning when angular input is within three standard deviations of mean
A = INtest(1,:,:)*180/pi;% angular input in degrees
A(A==0) = [];% remove the many 0 elements 
stdangularinput = std(A(:));

eps = 10^-4;% add epsilon to largest element of anglegrid so when angle equals max(anglegrid) then iangle is not empty
minangularinput = min(min(INtest(1,:,:))) * 180/pi; maxangularinput = max(max(INtest(1,:,:))) * 180/pi;% dangle = (maxangularinput - minangularinput)/100; anglegrid = [minangularinput:dangle:(maxangularinput-dangle) maxangularinput+eps];% 1 x something matrix, discretize the angular input 
minfortuning = max(-3*stdangularinput,minangularinput); maxfortuning = min(3*stdangularinput,maxangularinput);% if 3*stdangularinput is outside range of input then use min/max of input
dangle = (maxfortuning - minfortuning)/100; anglegrid = [minfortuning:dangle:(maxfortuning-dangle) maxfortuning+eps];% 1 x something matrix, discretize the angular input
maxminusmin_angularinput_preferred = -700*ones(numh,1);% strength of tuning to angular input at preferred integrated-angle
maxminusmin_angularinput_preferredfromRNNoutput = -700*ones(numh,1);% strength of tuning to angular input at preferred integrated-angle
slope_firingratevsangularinput_preferred = -700*ones(numh,1);% slope when a linear fit is made to the firing rate versus angular input curve
slope_firingratevsangularinput_preferredfromRNNoutput = -700*ones(numh,1);% slope when a linear fit is made to the firing rate versus angular input curve
firingrate_store = -700*ones(numh,numel(anglegrid)-1);% integrated angle is from ground truth when integrating inputs (integrated angle is used to calculate preferred integrated angle)
firingrate_preferredfromRNNoutput_store = -700*ones(numh,numel(anglegrid)-1);% integrated angle from RNN outputs (integrated angle is used to calculate preferred integrated angle)

% tuning curve at preferred integrated angle (integrated angle is ground truth from integrating inputs) for the whole range of input angles, not just when angular input is within three standard deviations of mean
minangularinputALL = min(min(INtest(1,:,:))) * 180/pi; maxangularinputALL = max(max(INtest(1,:,:))) * 180/pi; dangleALL = (maxangularinputALL - minangularinputALL)/100; anglegridALL = [minangularinputALL:dangleALL:(maxangularinputALL-dangleALL) maxangularinputALL+eps];% 1 x something matrix, discretize the angular input
firingrateALL_store = -700*ones(numh,numel(anglegridALL)-1);
firingratepredictions = cell(numh,1);
firingratepredictions_preferredfromRNNoutput = cell(numh,1);
for iunit=1:numh 
%for iunit=[75]    
    firingrate = zeros(1,numel(anglegrid)-1);% integrated angle is from ground truth when integrating inputs (integrated angle is used to calculate preferred integrated angle)
    firingrate_preferredfromRNNoutput = zeros(1,numel(anglegrid)-1);% integrated angle from RNN outputs (integrated angle is used to calculate preferred integrated angle)
    firingrateALL = zeros(1,numel(anglegridALL)-1);% integrated angle is from ground truth when integrating inputs (integrated angle is used to calculate preferred integrated angle)
    numvisits = zeros(1,numel(anglegrid)-1);% number of times RNN receives a specific angular input
    numvisits_preferredfromRNNoutput = zeros(1,numel(anglegrid)-1);% number of times RNN receives a specific angular input
    numvisitsALL = zeros(1,numel(anglegridALL)-1);% number of times RNN receives a specific angular input
    for itrial=1:numtrials
        RNNoutputangle = -700*ones(1,numTtest);
        RNNoutputangle(:) = atan2(y(1,:,itrial),y(2,:,itrial))*180/pi;% angle between -180 and +180 degrees
        iswitch = find(RNNoutputangle < 0); RNNoutputangle(iswitch) = 360 + RNNoutputangle(iswitch);% angle between 0 and 360 degrees
        for t=1:numTtest
            angleintegrated = angle_radians(t,itrial)*180/pi;% stored angle in degrees, between 0 and 360
            angleintegratedfromRNNoutput = RNNoutputangle(t);% stored angle in degrees, between 0 and 360
             
            if ((anglepreferred_integratedangularinput(iunit) - 1.5) <= angleintegrated) && (angleintegrated <= anglepreferred_integratedangularinput(iunit) + 1.5)
                angle = INtest(1,t,itrial)*180/pi;% angular input in degrees
                iposition = find(angle < anglegridALL ,1,'first') - 1;
                numvisitsALL(iposition) = numvisitsALL(iposition) + 1;
                firingrateALL(iposition) = firingrateALL(iposition) + h(iunit,t,itrial);
                
                if (minfortuning <= angle) && (angle <= maxfortuning)
                    iposition = find(angle < anglegrid ,1,'first') - 1;
                    numvisits(iposition) = numvisits(iposition) + 1;
                    firingrate(iposition) = firingrate(iposition) + h(iunit,t,itrial);
                end
            end
            if ((anglepreferred_integratedangularinputfromRNNoutput(iunit) - 1.5) <= angleintegratedfromRNNoutput) && (angleintegratedfromRNNoutput <= anglepreferred_integratedangularinputfromRNNoutput(iunit) + 1.5)
                angle = INtest(1,t,itrial)*180/pi;% angular input in degrees
                if (minfortuning <= angle) && (angle <= maxfortuning)
                    iposition = find(angle < anglegrid ,1,'first') - 1;
                    numvisits_preferredfromRNNoutput(iposition) = numvisits_preferredfromRNNoutput(iposition) + 1;
                    firingrate_preferredfromRNNoutput(iposition) = firingrate_preferredfromRNNoutput(iposition) + h(iunit,t,itrial);
                end
            end
        end% for t=1:numTtest
    end% for itrial=1:numtrials
    firingrate = firingrate ./ numvisits;
    firingrate_preferredfromRNNoutput = firingrate_preferredfromRNNoutput ./ numvisits_preferredfromRNNoutput;
    firingrateALL = firingrateALL ./ numvisitsALL;
    firingrate_store(iunit,:) = firingrate;
    firingrate_preferredfromRNNoutput_store(iunit,:) = firingrate_preferredfromRNNoutput;
    firingrateALL_store(iunit,:) = firingrateALL;
    maxminusmin_angularinput_preferred(iunit) = max(firingrate) - min(firingrate);% strength of tuning to angular input
    maxminusmin_angularinput_preferredfromRNNoutput(iunit) = max(firingrate_preferredfromRNNoutput) - min(firingrate_preferredfromRNNoutput);% strength of tuning to angular input
    
    for iteration=1:2
        % Determine the slope with a linear fit, don't fit portion of curve near 0, if entire curve is near 0 then fit this
        % Fit the target output y_=firingrate by linearly weighting regressors x_=anglegrid(1:end-1)+dangle/2
        % if numvisits is 0 then firingrate is NaN, remove NaN before linear regression
        x_ = [anglegrid(1:end-1)+dangle/2]';% numtrainingexamples x 1 matrix
        if iteration==1; y_ = firingrate'; end% numtrainingexamples x 1 matrix, noisy data
        if iteration==2; y_ = firingrate_preferredfromRNNoutput'; end% numtrainingexamples x 1 matrix, noisy data
        % remove firing rate when angular input is 0, there is a weird bump
        i0 = find(0 < anglegrid,1,'first') - 1;% this index is only good if y_ = firingrate, i.e. if y_ hasn't been changed by removing some elements
        x_(i0) = [];% remove 0
        y_(i0) = [];% remove 0
        inan = find(isnan(y_));
        x_(inan) = [];% remove NaN
        y_(inan) = [];% remove NaN
        iclose1 = find(y_>=0.95);  
        iclose0 = find(y_<0.05);
        %if (numel(y_)-numel(iclose0)-numel(iclose1)) > numel(y_)/4; iremove = [iclose0; iclose1]; x_(iremove) = []; y_(iremove) = []; end% if more than one quarter of curve is away from 0 or 1 then fit this portion after removing curve near 0 and 1
        if (numel(y_)-numel(iclose0)) > numel(y_)/4; iremove = iclose0; x_(iremove) = []; y_(iremove) = []; end% if more than one quarter of curve is away from 0 then fit this portion after removing curve near 0 
        numtrainingexamples = length(x_);
        X = [ones(numtrainingexamples,1) x_];% numtrainingexamples x 2 matrix
        %w = (X'*X)\X'*y;% inv(X'*X)*X'*y, w that minimizes the squared error between y and X*w
        w = pinv(X)*y_;% w that minimizes the squared error between y and X*w and has the smallest norm w'*w
        slope = w(2);
        firingrate_predictions_xdata = x_;
        firingrate_predictions_ydata = X*w;% numtrainingexamples x 1 matrix
       
        
        if iteration==1; slope_firingratevsangularinput_preferred(iunit) = slope; firingratepredictions{iunit}.xdata = firingrate_predictions_xdata; firingratepredictions{iunit}.ydata = firingrate_predictions_ydata; end
        if iteration==2; slope_firingratevsangularinput_preferredfromRNNoutput(iunit) = slope; firingratepredictions_preferredfromRNNoutput{iunit}.xdata = firingrate_predictions_xdata; firingratepredictions_preferredfromRNNoutput{iunit}.ydata = firingrate_predictions_ydata;end
    end
    %display(sprintf('unit %g',iunit))
end% for iunit=1:numh
save(fullfile(figuredir,'slope_firingratevsangularinput_preferred.mat'),'slope_firingratevsangularinput_preferred')
save(fullfile(figuredir,'slope_firingratevsangularinput_preferredfromRNNoutput.mat'),'slope_firingratevsangularinput_preferredfromRNNoutput')
save(fullfile(figuredir,'maxminusmin_angularinput_preferred.mat'),'maxminusmin_angularinput_preferred')
%load(fullfile(datadir,'slope_firingratevsangularinput_preferred.mat'))

% for iteration 4 sort units by compass, shiftpos, shiftneg, weakly tuned, remove
if exist(fullfile(datadir,'iunits_compass_sort.mat'),'file')==2; load(fullfile(datadir,'iunits_compass_sort.mat')); end
if exist(fullfile(datadir,'iunits_shiftpos_sort.mat'),'file')==2; load(fullfile(datadir,'iunits_shiftpos_sort.mat')); end
if exist(fullfile(datadir,'iunits_shiftneg_sort.mat'),'file')==2; load(fullfile(datadir,'iunits_shiftneg_sort.mat')); end
if exist(fullfile(datadir,'iunits_weaklytuned_sort.mat'),'file')==2; load(fullfile(datadir,'iunits_weaklytuned_sort.mat')); end
if exist(fullfile(datadir,'iunits_remove.mat'),'file')==2; load(fullfile(datadir,'iunits_remove.mat')); end
iteration_set = [1:3];
if exist('iunits_compass_sort','var') && exist('iunits_shiftpos_sort','var') && exist('iunits_shiftneg_sort','var') && exist('iunits_weaklytuned_sort','var') && exist('iunits_remove','var')
    desiredorder = [iunits_compass_sort; iunits_shiftpos_sort; iunits_shiftneg_sort; iunits_weaklytuned_sort; iunits_remove];
    if ~isequal(sort(desiredorder),[1:numh]'); error('missing units, line 1500'); end
    iteration_set = [1:4];
end
for iteration=iteration_set
    handle = figure;% firing of all hidden units as a function of the angular input (at the preferred integrated angle for each unit)
    clf; hold on; fontsize = 12; set(gcf,'DefaultLineLineWidth',4,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
    for iunit=1:numh
        if iteration==1; xplot = anglegrid(1:end-1)+dangle/2;        yplot = firingrate_store(iunit,:); end% integrated angle is from ground truth when integrating inputs (integrated angle is used to calculate preferred integrated angle)
        if iteration==2; xplot = anglegrid(1:end-1)+dangle/2;        yplot = firingrate_preferredfromRNNoutput_store(iunit,:); end% integrated angle from RNN outputs (integrated angle is used to calculate preferred integrated angle)
        if iteration==3; xplot = anglegridALL(1:end-1)+dangleALL/2;  yplot = firingrateALL_store(iunit,:); end% tuning curve at preferred integrated angle (integrated angle is ground truth from integrating inputs) for the whole range of input angles, not just when angular input is within three standard deviations of mean
        if iteration==4; xplot = anglegrid(1:end-1)+dangle/2;        yplot = firingrate_store(iunit,:); end% integrated angle is from ground truth when integrating inputs (integrated angle is used to calculate preferred integrated angle)
        inan = find(isnan(yplot)); xplot(inan) = []; yplot(inan) = [];% remove NaN
        
        numrowsinfigure = ceil(sqrt(numh));
        numcolumnsinfigure = ceil(numh/numrowsinfigure);
        positionofunit = iunit;% unit iunit is plotted at position positionofunit
        if iteration==4; positionofunit = find(desiredorder == iunit); end% unit iunit is plotted at position positionofunit
        subplot(numrowsinfigure,numcolumnsinfigure,positionofunit); hold on;
        plot(xplot,yplot,'k-')
        % plot firing rate predictions
        if iteration==1; plot(firingratepredictions{iunit}.xdata,firingratepredictions{iunit}.ydata,'r-','linewidth',1); end
        if iteration==2; plot(firingratepredictions_preferredfromRNNoutput{iunit}.xdata,firingratepredictions_preferredfromRNNoutput{iunit}.ydata,'r-','linewidth',1); end
        if iteration==3; plot(firingratepredictions{iunit}.xdata,firingratepredictions{iunit}.ydata,'r-','linewidth',1); end
        
        if iteration==1; xlim([min(anglegrid) max(anglegrid)]); end
        if iteration==2; xlim([min(anglegrid) max(anglegrid)]); end
        if iteration==3; xlim([min(anglegridALL) max(anglegridALL)]); end
        if iteration==4; xlim([min(anglegrid) max(anglegrid)]); end
        if isequal(nonlinearity{1},'tanh'); ylim([-1 1]); end
        if isequal(nonlinearity{1},'retanh'); ylim([0 1]); end
        if isequal(nonlinearity{1},'logistic'); ylim([0 1]); end
        if iteration==3 
            plot(minfortuning*ones(1,100),linspace(min(ylim),max(ylim),100),'k--','linewidth',1) 
            plot(maxfortuning*ones(1,100),linspace(min(ylim),max(ylim),100),'k--','linewidth',1) 
        end
        set(gca,'xtick',[],'ytick',[]);
        if iteration==1; title(sprintf('slope = %.2g, %g%s',slope_firingratevsangularinput_preferred(iunit),anglepreferred_integratedangularinput(iunit),char(176)),'fontsize',2,'fontweight','normal'); end% ax = gca; ax.FontSize = 1;
        if iteration==2; title(sprintf('slope = %.2g, %g%s',slope_firingratevsangularinput_preferredfromRNNoutput(iunit),anglepreferred_integratedangularinputfromRNNoutput(iunit),char(176)),'fontsize',2,'fontweight','normal'); end% ax = gca; ax.FontSize = 1;
        if iteration==3; title(sprintf('slope = %.2g, %g%s',slope_firingratevsangularinput_preferred(iunit),anglepreferred_integratedangularinput(iunit),char(176)),'fontsize',2,'fontweight','normal'); end% ax = gca; ax.FontSize = 1;
        if iteration==4; title(sprintf('slope = %.2g',slope_firingratevsangularinput_preferred(iunit)),'fontsize',2,'fontweight','normal'); end% ax = gca; ax.FontSize = 1;
        drawnow
        %pause
        %if rem(iunit,round(numh/10))==1; handlewaitbar = waitbar(iunit/numh); end% waitbar is very slow, redraw bar a maximum of ~10 times
    end% for iunit=1:numh
    %if exist('handlewaitbar','var'); close(handlewaitbar); clear handlewaitbar; end
    if iteration==1; [ax1,h1]=suplabel(sprintf('Angular input to RNN (%.3g to %.3g degrees)',min(anglegrid),max(anglegrid)),'x'); end
    if iteration==2; [ax1,h1]=suplabel(sprintf('Angular input to RNN (%.3g to %.3g degrees)',min(anglegrid),max(anglegrid)),'x'); end
    if iteration==3; [ax1,h1]=suplabel(sprintf('Angular input to RNN (%.3g to %.3g degrees)',min(anglegridALL),max(anglegridALL)),'x'); end
    if iteration==4; [ax1,h1]=suplabel(sprintf('Angular input to RNN (%.3g to %.3g degrees)',min(anglegrid),max(anglegrid)),'x'); end
    if isequal(nonlinearity{1},'tanh'); [ax2,h2]=suplabel('Activity of unit (-1 to 1)','y'); end
    if isequal(nonlinearity{1},'retanh'); [ax2,h2]=suplabel('Activity of unit (0 to 1)','y'); end
    if isequal(nonlinearity{1},'logistic'); [ax2,h2]=suplabel('Activity of unit (0 to 1)','y'); end
    [a3, h3] = suplabel({[sprintf('Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)]},'t');
    set(h1,'fontsize',fontsize); set(h2,'fontsize',fontsize); set(h3,'fontsize',fontsize,'fontweight','normal')
    %set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
    if iteration==1; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinput_epoch%g_numT%g_atpreferredintegratedangle',figuredir,epoch,numTtest)); end
    if iteration==2; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinput_epoch%g_numT%g_atpreferredintegratedanglefromRNNoutput',figuredir,epoch,numTtest)); end
    if iteration==3; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinput_epoch%g_numT%g_atpreferredintegratedangle_entireinputrange',figuredir,epoch,numTtest)); end
    if iteration==4; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinput_epoch%g_numT%g_atpreferredintegratedangle_sortcompassshiftposshiftneg',figuredir,epoch,numTtest)); end
end
clear firingratepredictions firingratepredictions_preferredfromRNNoutput% save memory


% distribution of slopes, remove nonresponsive units
itrialskeep = [1:numtrials]; 
itimekeep = [angle0duration+1:numTtest];
iunitsremove = [];% remove units that have ~constant activity
epsilon = 0.01;
for i=1:numh
    A = h(i,itimekeep,itrialskeep);
    %if numel(unique(A(:)))==1; iunitsremove = [iunitsremove; i]; end
    if (max(A(:)) - min(A(:))) < epsilon; iunitsremove = [iunitsremove; i]; end
end
iunitsremove = union(iunitsremove,find(maxminusmin_integratedangularinput < 0.03));% remove units that have very little modulation for integrated angular input
iunitskeep = [1:numh]'; iunitskeep(iunitsremove) = []; if ~isequal([1:numh]',sort([iunitskeep; iunitsremove])); error('missing units'); end
numhkeep = numel(iunitskeep);% number of units that do not have ~constant activity
% remove nonresponsive units
slope_firingratevsangularinput_preferred_keep = slope_firingratevsangularinput_preferred(iunitskeep);

dslope = 0.002;% round slope to nearest 0.001
x = round(slope_firingratevsangularinput_preferred_keep/dslope)*dslope;% numh x 1 matrix, find the unique elements of a vector x and how many times each unique element of x occurs
x = sort(x(:));
difference = diff([x;max(x)+1]); 
uniqueelements = x(difference~=0);
count = diff(find([1;difference]));

handle = figure;% plot distribution of slopes
clf; hold on; fontsize = 22; set(gcf,'DefaultLineLineWidth',8,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)  
bar(uniqueelements,count,'BarWidth',0.8,'FaceColor','b','EdgeColor','b') 
xlabel('Slope (firing rate/degree)')
ylabel('Number of units')
title({[sprintf('Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)];[sprintf('abs median/mean of %g units that do not have constant activity',numel(slope_firingratevsangularinput_preferred_keep))];[sprintf('= %g/%g',median(abs(slope_firingratevsangularinput_preferred_keep)),mean(abs(slope_firingratevsangularinput_preferred_keep)))]})
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
set(gca,'linewidth',2); set(gca,'TickLength',[0.02 0.025])% default set(gca,'TickLength',[0.01 0.025])
print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinput_epoch%g_numT%g_atpreferredintegratedangle_slopes',figuredir,epoch,numTtest))
   




%%
%--------------------------------------------------------------------------
%  classification bound for units (compass, shiftpos, shiftneg, weaklytuned, remove/dead)
%  strength of tuning for head direction (integrated angular input) is max - min response
%  strength of tuning for angular input is slope of tuning curve (at preferred head direction)
%--------------------------------------------------------------------------
% place classification bounds at top so code at bottom of rnn.m can be copy and pasted between different RNN models (bounds may change, but other analyses are the same)
SLOPECUTOFFmax_compass = 0.0175;% if units have slope above 0.02 they are classified as units that shift integrated angle positive
SLOPECUTOFFmin_compass = -0.0175;% if units have slope less than -0.02 they are classified as units that shift integrated angle negative
MAXMINUSMINCUTOFFmin_compass = 0.735;% if units have maximum minus minimum response for integrated angular input greater than MAXMINUSMINCUTOFFmin_compass (and unit is not strongly tuned for angular input, i.e. slope is between SLOPECUTOFFmax_compass and SLOPECUTOFFmin_compass) then units are compass units
SLOPECUTOFFmax_weaklytuned = 0.0065;% if units have slope above 0.02 they are classified as units that shift integrated angle positive
SLOPECUTOFFmin_weaklytuned = -0.0065;% if units have slope less than -0.02 they are classified as units that shift integrated angle negative
MAXMINUSMINCUTOFFmax_weaklytuned = 0.05;% if units have maximum minus minimum response for integrated angular input greater than MAXMINUSMINCUTOFFmin_compass (and unit is not strongly tuned for angular input, i.e. slope is between SLOPECUTOFFmax_compass and SLOPECUTOFFmin_compass) then units are compass units


if ~exist('maxminusmin_angularinput_preferred','var'); load(fullfile(figuredir,'maxminusmin_angularinput_preferred.mat')); end
if ~exist('maxminusmin_integratedangularinput','var'); load(fullfile(figuredir,'maxminusmin_integratedangularinput.mat')); end
if ~exist('maxminusmin_integratedangularinputfromRNNoutput','var'); load(fullfile(figuredir,'maxminusmin_integratedangularinputfromRNNoutput.mat')); end
if ~exist('slope_firingratevsangularinput_preferred','var'); load(fullfile(figuredir,'slope_firingratevsangularinput_preferred.mat')); end
if ~exist('slope_firingratevsangularinput_preferredfromRNNoutput','var'); load(fullfile(figuredir,'slope_firingratevsangularinput_preferredfromRNNoutput.mat')); end
if ~exist('anglepreferred_integratedangularinput','var'); load(fullfile(figuredir,'anglepreferred_integratedangularinput.mat')); end
if ~exist('anglepreferred_integratedangularinputfromRNNoutput','var'); load(fullfile(figuredir,'anglepreferred_integratedangularinputfromRNNoutput.mat')); end

for iteration=1:2   
    handle = figure;% scatter plot showing strength of tuning to integrated angular input VS angular input
    clf; hold on; fontsize = 15; set(gcf,'DefaultLineLineWidth',4,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
    pointsize = 70;
    if iteration==1
        z = anglepreferred_integratedangularinput;% color 2D scatterplot according to height of z, integrated angle that yields maximum firing, technically maximum absolute deviation from the mean (include this to better capture units that have high baseline firing and dip down for preferred angle)
        scatter(maxminusmin_integratedangularinput,maxminusmin_angularinput_preferred,pointsize,z,'filled'); 
    end
    if iteration==2 
        z = anglepreferred_integratedangularinput;% color 2D scatterplot according to height of z, integrated angle that yields maximum firing, technically maximum absolute deviation from the mean (include this to better capture units that have high baseline firing and dip down for preferred angle)
        scatter(maxminusmin_integratedangularinput,slope_firingratevsangularinput_preferred,pointsize,z,'filled') 
        plot(linspace(MAXMINUSMINCUTOFFmin_compass,max(maxminusmin_integratedangularinput),100),SLOPECUTOFFmax_compass*ones(1,100),'k-','linewidth',1)% selection box for compass units
        plot(linspace(MAXMINUSMINCUTOFFmin_compass,max(maxminusmin_integratedangularinput),100),SLOPECUTOFFmin_compass*ones(1,100),'k-','linewidth',1)% selection box for compass units
        plot(MAXMINUSMINCUTOFFmin_compass*ones(100,1),linspace(SLOPECUTOFFmin_compass,SLOPECUTOFFmax_compass,100),'k-','linewidth',1)% compass selection box
        plot(linspace(min(maxminusmin_integratedangularinput),MAXMINUSMINCUTOFFmax_weaklytuned,100),SLOPECUTOFFmax_weaklytuned*ones(1,100),'k-','linewidth',1)% selection box for weakly tuned units
        plot(linspace(min(maxminusmin_integratedangularinput),MAXMINUSMINCUTOFFmax_weaklytuned,100),SLOPECUTOFFmin_weaklytuned*ones(1,100),'k-','linewidth',1)% selection box for weakly tuned units
        plot(MAXMINUSMINCUTOFFmax_weaklytuned*ones(100,1),linspace(SLOPECUTOFFmin_weaklytuned,SLOPECUTOFFmax_weaklytuned,100),'k-','linewidth',1)% selection box for weakly tuned units
    end
    if BOUNDARY.periodic == 1; colormap(colorcet('C3')); caxis([0 360]); end% C3 and C7 are nice, circular colormap
    if BOUNDARY.periodic == 0; colormap(parula); end
    handle_colorbar = colorbar('FontSize',fontsize,'ytick',[0 180 360]); set(get(handle_colorbar,'ylabel'),'string','Preferred integrated angular input','FontSize',fontsize);% colormap(colors)% hsv, jet
    xlabel('Strength of tuning for integrated angular input (max-min response)')
    if iteration==1; ylabel('Strength of tuning for angular input at preferred integrated angle (max-min response)'); end
    if iteration==2; ylabel('Strength of tuning for angular input at preferred integrated angle (slope)'); end
    title({[sprintf('Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)]})
    if iteration==1; axis equal; end
    set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
    if iteration==1; print(handle, '-dpdf', sprintf('%s/allunits_strengthoftuning_epoch%g_numT%g_atpreferredintegratedangle',figuredir,epoch,numTtest)); end
    if iteration==2; print(handle, '-dpdf', sprintf('%s/allunits_strengthoftuning_epoch%g_numT%g_atpreferredintegratedangle_',figuredir,epoch,numTtest)); end
end


%--------------------------------------------------------------------------
%                   classify and sort units
%                         sort Whh 
%--------------------------------------------------------------------------
% Sort units into three groups based on tuning to 
% 1) integrated-angle 
% 2) shift integrated angle positive 
% 3) shift integrated angle negative
% Within groups sort units according to preferred integrated angle
% Sort elements of Whh by groups 1,2,3

iunits_remove = [];% remove units that have ~constant activity
itrialskeep = [1:numtrials]; 
itimekeep = [angle0duration+1:numTtest];
epsilon = 0.01;
for i=1:numh
    A = h(i,itimekeep,itrialskeep);
    %if numel(unique(A(:)))==1; iunits_remove = [iunits_remove; i]; end
    if (max(A(:)) - min(A(:))) < epsilon; iunits_remove = [iunits_remove; i]; end
end
iunits_remove = union(iunits_remove,find(maxminusmin_integratedangularinput < 0.03));% remove units that have very little modulation for integrated angular input
iunitskeep = [1:numh]'; iunitskeep(iunits_remove) = []; if ~isequal([1:numh]',sort([iunitskeep; iunits_remove])); error('missing units'); end
save(fullfile(figuredir,'iunits_remove.mat'),'iunits_remove');

iunits_posALL = find(slope_firingratevsangularinput_preferred > 0); iunits_posALL(ismember(iunits_posALL,iunits_remove)) = [];
iunits_negALL = find(slope_firingratevsangularinput_preferred < 0); iunits_negALL(ismember(iunits_negALL,iunits_remove)) = [];
if ~isequal([1:numh]',sort([iunits_remove; iunits_posALL; iunits_negALL])); error('missing trials posALL,negALL,remove'); end
save(fullfile(figuredir,'iunits_posALL.mat'),'iunits_posALL');
save(fullfile(figuredir,'iunits_negALL.mat'),'iunits_negALL');
% sort units based on preferred integrated-angle
anglepreferred_integratedangularinput_keep = anglepreferred_integratedangularinput(iunits_posALL);
[sorted, indices] = sort(anglepreferred_integratedangularinput_keep);
iunits_posALL_sort = iunits_posALL(indices);
% sort units based on preferred integrated-angle
anglepreferred_integratedangularinput_keep = anglepreferred_integratedangularinput(iunits_negALL);
[sorted, indices] = sort(anglepreferred_integratedangularinput_keep);
iunits_negALL_sort = iunits_negALL(indices);
if ~isequal([1:numh]',sort([iunits_remove; iunits_posALL_sort; iunits_negALL_sort])); error('missing trials posALL_sort,negALL_sort,remove'); end
save(fullfile(figuredir,'iunits_posALL_sort.mat'),'iunits_posALL_sort');
save(fullfile(figuredir,'iunits_negALL_sort.mat'),'iunits_negALL_sort');


iunits_compass = find((slope_firingratevsangularinput_preferred <= SLOPECUTOFFmax_compass) & (slope_firingratevsangularinput_preferred >= SLOPECUTOFFmin_compass) & (maxminusmin_integratedangularinput > MAXMINUSMINCUTOFFmin_compass));
iunits_weaklytuned = find((slope_firingratevsangularinput_preferred <= SLOPECUTOFFmax_weaklytuned) & (slope_firingratevsangularinput_preferred >= SLOPECUTOFFmin_weaklytuned) & (maxminusmin_integratedangularinput < MAXMINUSMINCUTOFFmax_weaklytuned));
iunits_weaklytuned(ismember(iunits_weaklytuned,iunits_remove)) = [];
iunits_shiftpos = iunits_posALL; iunits_shiftpos(ismember(iunits_shiftpos,[iunits_compass; iunits_weaklytuned])) = [];% shiftpos units are all positively tuned units that are not in compass and weakly tuned (and have some modulation)
iunits_shiftneg = iunits_negALL; iunits_shiftneg(ismember(iunits_shiftneg,[iunits_compass; iunits_weaklytuned])) = [];% shiftneg units are all negatively tuned units that are not in compass and weakly tuned (and have some modulation)
if ~isequal([1:numh]',sort([iunits_remove; iunits_shiftpos; iunits_shiftneg; iunits_compass; iunits_weaklytuned])); error('missing trials'); end

% define units when integrated angle is calculated from RNN outputs
iunits_compass_integratedanglefromRNNoutput = find((slope_firingratevsangularinput_preferredfromRNNoutput <= SLOPECUTOFFmax_compass) & (slope_firingratevsangularinput_preferredfromRNNoutput >= SLOPECUTOFFmin_compass) & (maxminusmin_integratedangularinputfromRNNoutput > MAXMINUSMINCUTOFFmin_compass));
iunits_posALL_integratedanglefromRNNoutput = find(slope_firingratevsangularinput_preferredfromRNNoutput > 0); iunits_posALL_integratedanglefromRNNoutput(ismember(iunits_posALL_integratedanglefromRNNoutput,iunits_remove)) = [];
iunits_negALL_integratedanglefromRNNoutput = find(slope_firingratevsangularinput_preferredfromRNNoutput < 0); iunits_negALL_integratedanglefromRNNoutput(ismember(iunits_negALL_integratedanglefromRNNoutput,iunits_remove)) = [];
iunits_weaklytuned_integratedanglefromRNNoutput = find((slope_firingratevsangularinput_preferredfromRNNoutput <= SLOPECUTOFFmax_weaklytuned) & (slope_firingratevsangularinput_preferredfromRNNoutput >= SLOPECUTOFFmin_weaklytuned) & (maxminusmin_integratedangularinputfromRNNoutput < MAXMINUSMINCUTOFFmax_weaklytuned));
iunits_weaklytuned_integratedanglefromRNNoutput(ismember(iunits_weaklytuned_integratedanglefromRNNoutput,iunits_remove)) = [];
iunits_shiftpos_integratedanglefromRNNoutput = iunits_posALL_integratedanglefromRNNoutput; iunits_shiftpos_integratedanglefromRNNoutput(ismember(iunits_shiftpos_integratedanglefromRNNoutput,[iunits_compass_integratedanglefromRNNoutput; iunits_weaklytuned_integratedanglefromRNNoutput])) = [];% shiftpos units are all positively tuned units that are not in compass and weakly tuned (and have some modulation)
iunits_shiftneg_integratedanglefromRNNoutput = iunits_negALL_integratedanglefromRNNoutput; iunits_shiftneg_integratedanglefromRNNoutput(ismember(iunits_shiftneg_integratedanglefromRNNoutput,[iunits_compass_integratedanglefromRNNoutput; iunits_weaklytuned_integratedanglefromRNNoutput])) = [];% shiftneg units are all negatively tuned units that are not in compass and weakly tuned (and have some modulation)
if ~isequal([1:numh]',sort([iunits_remove; iunits_shiftpos_integratedanglefromRNNoutput; iunits_shiftneg_integratedanglefromRNNoutput; iunits_compass_integratedanglefromRNNoutput; iunits_weaklytuned_integratedanglefromRNNoutput])); error('missing trials'); end

%-----
numhkeep = numel(iunitskeep);% number of units that do not have ~constant activity
% remove nonresponsive units
Whh_keep = Whh(iunitskeep,iunitskeep);% no sorting, just removing nonresponsive units

% sort units based on preferred integrated-angle
anglepreferred_integratedangularinput_keep = anglepreferred_integratedangularinput(iunitskeep);
[sorted, indices] = sort(anglepreferred_integratedangularinput_keep);
Whh_keep_sort = Whh_keep(indices,indices);

% sort weakly tuned units  according to preferred integrated angle
anglepreferred_integratedangularinput_keep = anglepreferred_integratedangularinput(iunits_weaklytuned);% preferred integrated angles of weakly tuned units
[sorted, indices] = sort(anglepreferred_integratedangularinput_keep);
iunits_weaklytuned_sort = iunits_weaklytuned(indices);
% sort units in group 1 according to preferred integrated angle, compass units with pure tuning for integrated angle
anglepreferred_integratedangularinput_keep = anglepreferred_integratedangularinput(iunits_compass);% preferred integrated angles of units in group 1
[sorted, indices] = sort(anglepreferred_integratedangularinput_keep);
iunits_compass_sort = iunits_compass(indices);
% sort units in group 2 according to preferred integrated angle, shiftpos units
anglepreferred_integratedangularinput_keep = anglepreferred_integratedangularinput(iunits_shiftpos);% preferred integrated angles of units in group 2
[sorted, indices] = sort(anglepreferred_integratedangularinput_keep);
iunits_shiftpos_sort = iunits_shiftpos(indices);
% sort units in group 3 according to preferred integrated angle, shiftneg units
anglepreferred_integratedangularinput_keep = anglepreferred_integratedangularinput(iunits_shiftneg);% preferred integrated angles of units in group 3
[sorted, indices] = sort(anglepreferred_integratedangularinput_keep);
iunits_shiftneg_sort = iunits_shiftneg(indices);
% sort Whh by groups 1,2,3
indices = [iunits_compass_sort; iunits_shiftpos_sort; iunits_shiftneg_sort];
Whh_keep_sortbygroup = Whh(indices,indices);
numhwelltuned = numel(indices);
save(fullfile(figuredir,'iunits_weaklytuned.mat'),'iunits_weaklytuned');
save(fullfile(figuredir,'iunits_compass.mat'),'iunits_compass');
save(fullfile(figuredir,'iunits_shiftpos.mat'),'iunits_shiftpos');
save(fullfile(figuredir,'iunits_shiftneg.mat'),'iunits_shiftneg');
save(fullfile(figuredir,'iunits_weaklytuned_sort.mat'),'iunits_weaklytuned_sort');% sort units by preferred integrated-angle
save(fullfile(figuredir,'iunits_compass_sort.mat'),'iunits_compass_sort');% sort units by preferred integrated-angle
save(fullfile(figuredir,'iunits_shiftpos_sort.mat'),'iunits_shiftpos_sort');% sort units by preferred integrated-angle
save(fullfile(figuredir,'iunits_shiftneg_sort.mat'),'iunits_shiftneg_sort');% sort units by preferred integrated-angle

% sort units in group 1 according to preferred integrated angle (integrated angle computed from RNN output), compass units with pure tuning for integrated angle
anglepreferred_keep = anglepreferred_integratedangularinputfromRNNoutput(iunits_compass_integratedanglefromRNNoutput);% preferred integrated angles of units in group 1
[sorted, indices] = sort(anglepreferred_keep);
iunits_compass_integratedanglefromRNNoutput_sort = iunits_compass_integratedanglefromRNNoutput(indices);
% sort units in group 2 according to preferred integrated angle, shiftpos units
anglepreferred_keep = anglepreferred_integratedangularinputfromRNNoutput(iunits_shiftpos_integratedanglefromRNNoutput);% preferred integrated angles of units in group 2
[sorted, indices] = sort(anglepreferred_keep);
iunits_shiftpos_integratedanglefromRNNoutput_sort = iunits_shiftpos_integratedanglefromRNNoutput(indices);
% sort units in group 3 according to preferred integrated angle, shiftneg units
anglepreferred_keep = anglepreferred_integratedangularinputfromRNNoutput(iunits_shiftneg_integratedanglefromRNNoutput);% preferred integrated angles of units in group 3
[sorted, indices] = sort(anglepreferred_keep);
iunits_shiftneg_integratedanglefromRNNoutput_sort = iunits_shiftneg_integratedanglefromRNNoutput(indices);
% sort weakly tuned units according to preferred integrated angle
anglepreferred_keep = anglepreferred_integratedangularinputfromRNNoutput(iunits_weaklytuned_integratedanglefromRNNoutput);% preferred integrated angles of units in group 3
[sorted, indices] = sort(anglepreferred_keep);
iunits_weaklytuned_integratedanglefromRNNoutput_sort = iunits_weaklytuned_integratedanglefromRNNoutput(indices);
% sort Whh by groups 1,2,3
indices = [iunits_compass_integratedanglefromRNNoutput_sort; iunits_shiftpos_integratedanglefromRNNoutput_sort; iunits_shiftneg_integratedanglefromRNNoutput_sort];
Whh_keep_sortbygroup_integratedanglefromRNNoutput = Whh(indices,indices);
numhwelltuned_integratedanglefromRNNoutput = numel(indices);
save(fullfile(figuredir,'iunits_weaklytuned_integratedanglefromRNNoutput.mat'),'iunits_weaklytuned_integratedanglefromRNNoutput');
save(fullfile(figuredir,'iunits_compass_integratedanglefromRNNoutput.mat'),'iunits_compass_integratedanglefromRNNoutput');
save(fullfile(figuredir,'iunits_shiftpos_integratedanglefromRNNoutput.mat'),'iunits_shiftpos_integratedanglefromRNNoutput');
save(fullfile(figuredir,'iunits_shiftneg_integratedanglefromRNNoutput.mat'),'iunits_shiftneg_integratedanglefromRNNoutput');
save(fullfile(figuredir,'iunits_compass_integratedanglefromRNNoutput_sort.mat'),'iunits_compass_integratedanglefromRNNoutput_sort');% sort units by preferred integrated-angle
save(fullfile(figuredir,'iunits_shiftpos_integratedanglefromRNNoutput_sort.mat'),'iunits_shiftpos_integratedanglefromRNNoutput_sort');% sort units by preferred integrated-angle
save(fullfile(figuredir,'iunits_shiftneg_integratedanglefromRNNoutput_sort.mat'),'iunits_shiftneg_integratedanglefromRNNoutput_sort');% sort units by preferred integrated-angle
save(fullfile(figuredir,'iunits_weaklytuned_integratedanglefromRNNoutput_sort.mat'),'iunits_weaklytuned_integratedanglefromRNNoutput_sort');% sort units by preferred integrated-angle

for iteration=1:4
    handle = figure;% Whh sorted according to groups 1,2,3 and then anglepreferred_integratedangularinput within groups
    hold on; fontsize = 18; set(gcf,'DefaultLineLineWidth',4,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
    if iteration==1; imagesc([1:numhkeep],[1:numhkeep],Whh_keep); end
    if iteration==2; imagesc([1:numhkeep],[1:numhkeep],Whh_keep_sort); end
    if iteration==3
        imagesc([1:numhwelltuned],[1:numhwelltuned],Whh_keep_sortbygroup) 
        plot(1:numhwelltuned,(numel(iunits_compass_sort)+0.5)*ones(1,numhwelltuned),'k-','linewidth',2); plot((numel(iunits_compass_sort)+0.5)*ones(1,numhwelltuned),1:numhwelltuned,'k-','linewidth',2)% line at the boundary of group 1
        plot(1:numhwelltuned,(numel(iunits_compass_sort)+numel(iunits_shiftpos_sort)+0.5)*ones(1,numhwelltuned),'k-','linewidth',2); plot((numel(iunits_compass_sort)+numel(iunits_shiftpos_sort)+0.5)*ones(1,numhwelltuned),1:numhwelltuned,'k-','linewidth',2)% line at the boundary of group 2
    end% A(1,1) is in the top left corner of the image, if not set(gca,'YDir','reverse')
    if iteration==4
        imagesc([1:numhwelltuned_integratedanglefromRNNoutput],[1:numhwelltuned_integratedanglefromRNNoutput],Whh_keep_sortbygroup_integratedanglefromRNNoutput) 
        plot(1:numhwelltuned_integratedanglefromRNNoutput,(numel(iunits_compass_integratedanglefromRNNoutput_sort)+0.5)*ones(1,numhwelltuned_integratedanglefromRNNoutput),'k-','linewidth',2); plot((numel(iunits_compass_integratedanglefromRNNoutput_sort)+0.5)*ones(1,numhwelltuned_integratedanglefromRNNoutput),1:numhwelltuned_integratedanglefromRNNoutput,'k-','linewidth',2)% line at the boundary of group 1
        plot(1:numhwelltuned_integratedanglefromRNNoutput,(numel(iunits_compass_integratedanglefromRNNoutput_sort)+numel(iunits_shiftpos_integratedanglefromRNNoutput_sort)+0.5)*ones(1,numhwelltuned_integratedanglefromRNNoutput),'k-','linewidth',2); plot((numel(iunits_compass_integratedanglefromRNNoutput_sort)+numel(iunits_shiftpos_integratedanglefromRNNoutput_sort)+0.5)*ones(1,numhwelltuned_integratedanglefromRNNoutput),1:numhwelltuned_integratedanglefromRNNoutput,'k-','linewidth',2)% line at the boundary of group 2
    end% A(1,1) is in the top left corner of the image, if not set(gca,'YDir','reverse')
    set(gca,'YDir','reverse')
    if iteration==1; handle_colorbar = colorbar('FontSize',fontsize); set(get(handle_colorbar,'ylabel'),'string',{[sprintf('Whh after removing %g unresponsive units',numel(iunits_remove))]},'FontSize',fontsize); end% colormap(colors)% hsv, jet
    if iteration==2; handle_colorbar = colorbar('FontSize',fontsize); set(get(handle_colorbar,'ylabel'),'string',{[sprintf('Whh after removing %g unresponsive units',numel(iunits_remove))];['sorted by preferred integrated-angle']},'FontSize',fontsize); end% colormap(colors)% hsv, jet
    if iteration==3; handle_colorbar = colorbar('FontSize',fontsize); set(get(handle_colorbar,'ylabel'),'string',{[sprintf('Whh after removing %g unresponsive and %g weakly tuned units',numel(iunits_remove),numel(iunits_weaklytuned))];['sorted by preferred integrated-angle']},'FontSize',fontsize); end% colormap(colors)% hsv, jet
    if iteration==4; handle_colorbar = colorbar('FontSize',fontsize); set(get(handle_colorbar,'ylabel'),'string',{[sprintf('Whh after removing %g unresponsive and %g weakly tuned units',numel(iunits_remove),numel(iunits_weaklytuned_integratedanglefromRNNoutput))];['sorted by preferred integrated-angle']},'FontSize',fontsize); end% colormap(colors)% hsv, jet
    colormap(bluewhitered)% color so anything positive is red and anything negative is blue, also see https://www.mathworks.com/matlabcentral/answers/305073-colormap-fixed-middle-value
    xlabel('Unit')
    ylabel('Unit')
    if iteration==3; xlabel(sprintf('Unit (%g compass, %g shiftpos, %g shiftneg)',numel(iunits_compass),numel(iunits_shiftpos),numel(iunits_shiftneg))); ylabel(sprintf('Unit (%g compass, %g shiftpos, %g shiftneg)',numel(iunits_compass),numel(iunits_shiftpos),numel(iunits_shiftneg))); end
    if iteration==4; xlabel(sprintf('Unit (%g compass, %g shiftpos, %g shiftneg)',numel(iunits_compass_integratedanglefromRNNoutput),numel(iunits_shiftpos_integratedanglefromRNNoutput),numel(iunits_shiftneg_integratedanglefromRNNoutput))); ylabel(sprintf('Unit (%g compass, %g shiftpos, %g shiftneg)',numel(iunits_compass_integratedanglefromRNNoutput),numel(iunits_shiftpos_integratedanglefromRNNoutput),numel(iunits_shiftneg_integratedanglefromRNNoutput))); end
    title({[sprintf('Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)]})
    axis tight; axis equal
    set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
    if iteration==1; print(handle, '-dpdf', sprintf('%s/Whh_keep_epoch%g_numT%g',figuredir,epoch,numTtest)); end 
    if iteration==2; print(handle, '-dpdf', sprintf('%s/Whh_keep_sorted_epoch%g_numT%g',figuredir,epoch,numTtest)); end 
    if iteration==3; print(handle, '-dpdf', sprintf('%s/Whh_welltuned_sortedbygroup_epoch%g_numT%g',figuredir,epoch,numTtest)); end 
    if iteration==4; print(handle, '-dpdf', sprintf('%s/Whh_welltuned_sortedbygroup_integratedanglefromRNNoutput_epoch%g_numT%g',figuredir,epoch,numTtest)); end 
end



%% --------------------------------------------------------------------------
% connectivity strength (Whh) between units as a function of relative preferred integrated-angle
%--------------------------------------------------------------------------
close all
% test relative angle calculation, result should be between +180 and -180 degrees
if 0
    angle = 0; angle_reference = 359;% relativeangle = 1
    angle = 359; angle_reference = 0;% relativeangle = -1
    angle = 180; angle_reference = 0;% relativeangle = 180
    angle = 0; angle_reference = 179.9;% relativeangle = -179.9
    angle = 20; angle_reference = 0;% relativeangle = 20
    angle = 340; angle_reference = 0;% relativeangle = -20
    angle = 190; angle_reference = 180;% relativeangle = 10
    angle = 170; angle_reference = 180;% relativeangle = -10
    angle = 10; angle_reference = 350;% relativeangle = 20
    angle = 350; angle_reference = 10;% relativeangle = -20
    relativeangle = mod(angle - angle_reference,360);% a number between 0 and 359.9
    if relativeangle > 180; relativeangle = relativeangle - 360; end% a number between -179.9 and 180
    display(sprintf('angle = %g, angle_reference = %g, relativeangle = %g',angle,angle_reference,relativeangle))
end
for iteration=1:20    
    % code is currently written so dangle (the width of the bin over which connection weights are averaged) must be an odd number
    if iteration==1;  iunitsfrom = iunits_compass; iunitsto = iunits_compass; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'compass'; unitstolabel = 'compass'; figuresuffix = 'fromcompasstocompass'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==2;  iunitsfrom = iunits_compass; iunitsto = iunits_shiftpos; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'compass'; unitstolabel = 'shiftpos'; figuresuffix = 'fromcompasstoshiftpos'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==3;  iunitsfrom = iunits_compass; iunitsto = iunits_shiftneg; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'compass'; unitstolabel = 'shiftneg'; figuresuffix = 'fromcompasstoshiftneg'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==4;  iunitsfrom = iunits_shiftpos; iunitsto = iunits_compass; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'shiftpos'; unitstolabel = 'compass'; figuresuffix = 'fromshiftpostocompass'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==5;  iunitsfrom = iunits_shiftpos; iunitsto = iunits_shiftpos; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'shiftpos'; unitstolabel = 'shiftpos'; figuresuffix = 'fromshiftpostoshiftpos'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==6;  iunitsfrom = iunits_shiftpos; iunitsto = iunits_shiftneg; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'shiftpos'; unitstolabel = 'shiftneg'; figuresuffix = 'fromshiftpostoshiftneg'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==7;  iunitsfrom = iunits_shiftneg; iunitsto = iunits_compass; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'shiftneg'; unitstolabel = 'compass'; figuresuffix = 'fromshiftnegtocompass'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==8;  iunitsfrom = iunits_shiftneg; iunitsto = iunits_shiftpos; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'shiftneg'; unitstolabel = 'shiftpos'; figuresuffix = 'fromshiftnegtoshiftpos'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==9;  iunitsfrom = iunits_shiftneg; iunitsto = iunits_shiftneg; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'shiftneg'; unitstolabel = 'shiftneg'; figuresuffix = 'fromshiftnegtoshiftneg'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    
    if iteration==10;  iunitsfrom = iunits_compass_integratedanglefromRNNoutput; iunitsto = iunits_compass_integratedanglefromRNNoutput; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'compass'; unitstolabel = 'compass'; figuresuffix = 'fromcompasstocompass_integratedanglefromRNNoutput'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==11;  iunitsfrom = iunits_shiftpos_integratedanglefromRNNoutput; iunitsto = iunits_compass_integratedanglefromRNNoutput; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'shiftpos'; unitstolabel = 'compass'; figuresuffix = 'fromshiftpostocompass_integratedanglefromRNNoutput'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==12;  iunitsfrom = iunits_shiftneg_integratedanglefromRNNoutput; iunitsto = iunits_compass_integratedanglefromRNNoutput; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'shiftneg'; unitstolabel = 'compass'; figuresuffix = 'fromshiftnegtocompass_integratedanglefromRNNoutput'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==13;  iunitsfrom = iunits_compass_integratedanglefromRNNoutput; iunitsto = iunits_shiftpos_integratedanglefromRNNoutput; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'compass'; unitstolabel = 'shiftpos'; figuresuffix = 'fromcompasstoshiftpos_integratedanglefromRNNoutput'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==14;  iunitsfrom = iunits_compass_integratedanglefromRNNoutput; iunitsto = iunits_shiftneg_integratedanglefromRNNoutput; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'compass'; unitstolabel = 'shiftneg'; figuresuffix = 'fromcompasstoshiftneg_integratedanglefromRNNoutput'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==15;  iunitsfrom = iunits_shiftpos_integratedanglefromRNNoutput; iunitsto = iunits_shiftpos_integratedanglefromRNNoutput; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'shiftpos'; unitstolabel = 'shiftpos'; figuresuffix = 'fromshiftpostoshiftpos_integratedanglefromRNNoutput'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==16;  iunitsfrom = iunits_shiftneg_integratedanglefromRNNoutput; iunitsto = iunits_shiftneg_integratedanglefromRNNoutput; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'shiftneg'; unitstolabel = 'shiftneg'; figuresuffix = 'fromshiftnegtoshiftneg_integratedanglefromRNNoutput'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    
    if iteration==17;  iunitsfrom = iunits_posALL; iunitsto = iunits_posALL; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'posALL'; unitstolabel = 'posALL'; figuresuffix = 'fromposALLtoposALL'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==18;  iunitsfrom = iunits_posALL; iunitsto = iunits_negALL; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'posALL'; unitstolabel = 'negALL'; figuresuffix = 'fromposALLtonegALL'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==19;  iunitsfrom = iunits_negALL; iunitsto = iunits_negALL; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'negALL'; unitstolabel = 'negALL'; figuresuffix = 'fromnegALLtonegALL'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    if iteration==20;  iunitsfrom = iunits_negALL; iunitsto = iunits_posALL; dangle = 5; relativeanglegrid = [-180:1:180]; unitsfromlabel = 'negALL'; unitstolabel = 'posALL'; figuresuffix = 'fromnegALLtoposALL'; end% connectivity when preferred integrated-angle differs by amount in anglegrid
    
    %connectionstrength = NaN(numel(relativeanglegrid),numel(iunitsto),numel(iunitsfrom));% connection strength from unit iunitsfrom(i) to other units with varying degrees of similarity, where similarity is defined by preferred angle
    connectionstrength = NaN(numel(relativeanglegrid)-1,numel(iunitsto),numel(iunitsfrom));% CJC 2022.7.2 I think the above code is wrong and this is correct
    for i=1:numel(iunitsfrom)% connection strength from unit iunitsfrom(i) to unit iunitsto(j)
        for j=1:numel(iunitsto)
            % what is the relative preferred integrated-angle of unit iunitsto(j) relative to unit iunitsfrom(i)? A number between +180 and -180 degrees
            if iteration<=9; relativeangle = mod(anglepreferred_integratedangularinput(iunitsto(j)) - anglepreferred_integratedangularinput(iunitsfrom(i)),360); end% a number between 0 and 359.9
            if (10<=iteration) && (iteration<=16); relativeangle = mod(anglepreferred_integratedangularinputfromRNNoutput(iunitsto(j)) - anglepreferred_integratedangularinputfromRNNoutput(iunitsfrom(i)),360); end% a number between 0 and 359.9
            if (17<=iteration) && (iteration<=20); relativeangle = mod(anglepreferred_integratedangularinput(iunitsto(j)) - anglepreferred_integratedangularinput(iunitsfrom(i)),360); end% a number between 0 and 359.9
            if relativeangle > 180; relativeangle = relativeangle - 360; end% a number between -179.9 and 180

            %ibin = find((relativeanglegrid-1+10^-4 <= relativeangle) & (relativeangle <= relativeanglegrid));
            ibin = find(relativeanglegrid < relativeangle, 1, 'last');% CJC 2022.7.2 I think the above code is wrong and this is correct
            ibins = ibin-(dangle-1)/2:ibin+(dangle-1)/2;% dangle number of bins centered around ibin, code is written so dangle must be an odd number
            if numel(ibins)~=dangle; error('problem 1 with number of bins'); end
            %ilessthan1 = find(ibins<1); ibins(ilessthan1) = numel(relativeanglegrid)+ibins(ilessthan1);% wrap bins with indices <= 0
            %igreater = find(ibins>numel(relativeanglegrid)); ibins(igreater) = ibins(igreater) - numel(relativeanglegrid);% wrap bins with indices >= numel(relativeanglegrid)
            ilessthan1 = find(ibins<1); ibins(ilessthan1) = numel(relativeanglegrid)-1 + ibins(ilessthan1);% CJC 2022.7.2 I think the above code is wrong and this is correct
            igreater = find(ibins>(numel(relativeanglegrid)-1)); ibins(igreater) = ibins(igreater) - (numel(relativeanglegrid)-1);% CJC 2022.7.2 I think the above code is wrong and this is correct
            if numel(ibins)~=dangle; error('problem 2 with number of bins'); end
            if iunitsfrom(i) ~= iunitsto(j)% don't include self-connections, comment out to include self-connections
                connectionstrength(ibins,j,i) = Whh(iunitsto(j),iunitsfrom(i));% connection strength from unit iunitsfrom(i) to other units with varying degrees of similarity, where similarity is defined by preferred integrated-angle
                if sum(~isnan(connectionstrength(:,j,i)))~=dangle; error('problem 3 with number of bins'); end% number of nonnan elements
            end
        end
    end
    connectionstrength = connectionstrength(:,:);% numel(relativeanglegrid) x numel(iunitsto)*numel(iunitsfrom) matrix
    connectionstrength = connectionstrength';% numel(iunitsto)*numel(iunitsfrom) x numel(relativeanglegrid) matrix
    %sum(~isnan(connectionstrength),2)% number of nonnan elements in each row should be the same 
    for i=1:size(connectionstrength,1)% all the non-nan elements in each row should be the same
        inotnan = find(isnan(connectionstrength(i,:))==0);% indices that are not nan
        %if numel(inotnan)~=dangle; error('there should be dangle non-nan elements in each row'); end% this will not hold if we don't allow self-connections, rows storing weights for self-connections will be all NaN
        if numel(inotnan)>0 && numel(inotnan)~=dangle; error('there should be dangle non-nan elements in each row'); end% each non-nan row should have dangle entries, rows storing weights for self-connections will be all NaN
        if numel(inotnan)>0 && numel(unique(connectionstrength(i,inotnan)))~=1; error('all the non-nan elements in each row should be the same'); end% added numel(inotnan)>0 because if self-connections are not included then some rows will have all NaN and so inotnan is an empty matrix and numel(unique(connectionstrength(i,inotnan))) is 0
    end  
    
    handle = figure;
    hold on; fontsize = 24; set(gcf,'DefaultLineLineWidth',4,'DefaultLineMarkerSize',10,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)% only need to set DefaulttextFontSize if using function text
    %Tplot = relativeanglegrid-1/2;
    Tplot = relativeanglegrid(1:end-1)+1/2;% CJC 2022.7.2 I think the above code is wrong and this is correct
    Amean = nanmean(connectionstrength,1);% 1 x numel(relativeanglegrid)
    Astd = nanstd(connectionstrength,1,1);% 1 x numel(relativeanglegrid)
    inotnan = find(isnan(Amean)==0);% indices that are not nan
    Tplot = Tplot(inotnan); Amean = Amean(inotnan); Astd = Astd(inotnan);
    patch([Tplot fliplr(Tplot)], [Amean-Astd fliplr(Amean+Astd)], 'k', 'EdgeColor', 'w', 'EdgeAlpha', 0.4, 'FaceAlpha', 0.4);
    plot(Tplot,Amean,'k-')
    plot(Tplot,zeros(size(Tplot)),'k--','linewidth',1)
    set(gca, 'XTick',[min(relativeanglegrid):90:max(relativeanglegrid)]);
    xlabel(sprintf('Preferred integrated-angle relative to %s unit (degrees)',unitsfromlabel))
    ylabel('Mean connection strength (Whh)')
    axis tight; axis([-Inf Inf min(ylim)-abs(max(ylim)-min(ylim))/100  max(ylim)+abs(max(ylim)-min(ylim))/100])
    xlim([min(relativeanglegrid) max(relativeanglegrid)])
    title({[sprintf('From %g %s units to %g %s units',numel(iunitsfrom),unitsfromlabel,numel(iunitsto),unitstolabel)];[sprintf('Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%, %g%s bins slid by %g%s',normalizederror,dangle,char(176),unique(diff(relativeanglegrid)),char(176))]})
    set(gca,'linewidth',2); set(gca,'TickLength',[0.02 0.025])% default set(gca,'TickLength',[0.01 0.025])
    set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
    print(handle, '-dpdf', sprintf('%s/Whh_epoch%g_numT%g_dangle%g_%s_',figuredir,epoch,numTtest,dangle,figuresuffix))
    
    handle = figure;
    %subplot(2,1,2)
    hold on; fontsize = 15; set(gcf,'DefaultLineLineWidth',4,'DefaultLineMarkerSize',10,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)% only need to set DefaulttextFontSize if using function text
    %Tplot = relativeanglegrid-1/2;
    Tplot = relativeanglegrid(1:end-1)+1/2;% CJC 2022.7.2 I think the above code is wrong and this is correct
    colors = jet(numel(iunitsfrom)*numel(iunitsto)); colors(1,:) = [0 0 1]; colors(end,:) = [1 0 0];% colors(1,:) is blue, colors(end,:) is red
    for i=1:numel(iunitsfrom)*numel(iunitsto)
        inotnan = find(isnan(connectionstrength(i,:))==0);% indices that are not nan
        plot(Tplot(inotnan),connectionstrength(i,inotnan),'.','color',colors(i,:),'markersize',10)
    end
    plot(Tplot,zeros(size(Tplot)),'k--','linewidth',1)
    set(gca, 'XTick',[min(relativeanglegrid):45:max(relativeanglegrid)]);
    xlabel(sprintf('Preferred integrated-angle relative to %s unit (degrees)',unitsfromlabel))
    ylabel('Connection strength (Whh)')
    axis tight; axis([-Inf Inf min(ylim)-abs(max(ylim)-min(ylim))/100  max(ylim)+abs(max(ylim)-min(ylim))/100])
    xlim([min(relativeanglegrid) max(relativeanglegrid)])
    title({[sprintf('From %g %s units to %g %s units',numel(iunitsfrom),unitsfromlabel,numel(iunitsto),unitstolabel)];[sprintf('Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%, %g%s bins slid by %g%s',normalizederror,dangle,char(176),unique(diff(relativeanglegrid)),char(176))]})
    set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
    print(handle, '-dpdf', sprintf('%s/Whh_epoch%g_numT%g_dangle%g_%s',figuredir,epoch,numTtest,dangle,figuresuffix))
end% for iteration=1:20


% Do the preferred integrated-angles tile 0 to 360?
% Are the preferred integrated-angles evenly spaced?
%for iteration=3
for iteration=1:3
    handle = figure;% preferred integrated-angle for each unit (excluding units that have ~constant activity)
    clf; hold on; fontsize = 17; set(gcf,'DefaultLineLineWidth',8,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
    if iteration==1
        plot(1:numel(iunits_posALL),sort(anglepreferred_integratedangularinput(iunits_posALL)),'k-')
        plot(1:numel(iunits_negALL),sort(anglepreferred_integratedangularinput(iunits_negALL)),'r-')
        legend({[sprintf('%g posALL units',numel(iunits_posALL))];[sprintf('%g negALL units',numel(iunits_negALL))]},'location','southeast')
    end
    if iteration==2
        plot(1:numel(iunits_compass),sort(anglepreferred_integratedangularinput(iunits_compass)),'r-')
        plot(1:numel(iunits_shiftneg),sort(anglepreferred_integratedangularinput(iunits_shiftneg)),'g-')% clockwise
        plot(1:numel(iunits_shiftpos),sort(anglepreferred_integratedangularinput(iunits_shiftpos)),'b-')% counterclockwise
        plot(1:numel(iunits_weaklytuned),sort(anglepreferred_integratedangularinput(iunits_weaklytuned)),'k-')
        legend({[sprintf('%g compass units',numel(iunits_compass))];[sprintf('%g shiftneg units',numel(iunits_shiftneg))];[sprintf('%g shiftpos units',numel(iunits_shiftpos))];[sprintf('%g weakly tuned units',numel(iunits_weaklytuned))]},'location','southeast')
    end
    xlabel('Unit (sorted)')
    ylabel('Preferred integrated-angle (degrees)')
    title({[sprintf('Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)]})
    if iteration==3% same as 2 but larger font, no weaklytuned
        clf; hold on; fontsize = 26; set(gcf,'DefaultLineLineWidth',8,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
        plot(1:numel(iunits_compass),sort(anglepreferred_integratedangularinput(iunits_compass)),'r.')
        plot(1:numel(iunits_shiftneg),sort(anglepreferred_integratedangularinput(iunits_shiftneg)),'g.')% clockwise
        plot(1:numel(iunits_shiftpos),sort(anglepreferred_integratedangularinput(iunits_shiftpos)),'b.')% counterclockwise
        set(gca,'YTick',[0 180 360],'yTickLabel',[0 180 360])
        set(gca,'linewidth',2); set(gca,'TickLength',[0.02 0.025])% default set(gca,'TickLength',[0.01 0.025])
        title('')
    end
    set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
    axis tight; axis([-Inf Inf min(ylim)-abs(max(ylim)-min(ylim))/100  max(ylim)+abs(max(ylim)-min(ylim))/100])
    if iteration==1; print(handle, '-dpdf', sprintf('%s/allunits_preferredintegratedangularinput_epoch%g_numT%g_groupedposALLnegALL',figuredir,epoch,numTtest)); end
    if iteration==2; print(handle, '-dpdf', sprintf('%s/allunits_preferredintegratedangularinput_epoch%g_numT%g_groupedcompassshiftposshiftneg',figuredir,epoch,numTtest)); end
    if iteration==3; print(handle, '-dpdf', sprintf('%s/allunits_preferredintegratedangularinput_epoch%g_numT%g_groupedcompassshiftposshiftneg_',figuredir,epoch,numTtest)); end
end

% Plot preferred integrated-angles arranged around a circle
handle = figure;% preferred angle for each unit (excluding units that are weakly tuned), arranged around a circle
clf; hold on; fontsize = 10; set(gcf,'DefaultLineLineWidth',8,'DefaultLineMarkerSize',26,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
plot(cos(anglepreferred_integratedangularinput(iunits_compass)*pi/180),sin(anglepreferred_integratedangularinput(iunits_compass)*pi/180),'r.')
plot(cos(anglepreferred_integratedangularinput(iunits_shiftneg)*pi/180),sin(anglepreferred_integratedangularinput(iunits_shiftneg)*pi/180),'g.')% clockwise
plot(cos(anglepreferred_integratedangularinput(iunits_shiftpos)*pi/180),sin(anglepreferred_integratedangularinput(iunits_shiftpos)*pi/180),'b.')% counterclockwise
axis off
axis equal
title({[sprintf('Preferrred angles for %g/%g/%g compass/shiftneg(CW)/shiftpos(CCW) units',numel(iunits_compass),numel(iunits_shiftneg),numel(iunits_shiftpos))];[sprintf('Training epoch %g, %g trials, %g timesteps in simulation, normalized error overall = %.5g%%',epoch,numtrials,numTtest,normalizederror)];[]})
set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
print(handle, '-dpdf', sprintf('%s/allunits_preferredintegratedangularinput_epoch%g_numT%g_groupedcompassshiftposshiftneg_arrangedcircle',figuredir,epoch,numTtest));








%%
close all
if ~exist('iunits_compass','var'); load(fullfile(figuredir,'iunits_compass.mat')); end
if ~exist('iunits_compass_sort','var'); load(fullfile(figuredir,'iunits_compass_sort.mat')); end
if ~exist('iunits_shiftpos','var'); load(fullfile(figuredir,'iunits_shiftpos.mat')); end
if ~exist('iunits_shiftpos_sort','var'); load(fullfile(figuredir,'iunits_shiftpos_sort.mat')); end
if ~exist('iunits_shiftneg','var'); load(fullfile(figuredir,'iunits_shiftneg.mat')); end
if ~exist('iunits_shiftneg_sort','var'); load(fullfile(figuredir,'iunits_shiftneg_sort.mat')); end
if ~exist('iunits_weaklytuned','var'); load(fullfile(figuredir,'iunits_weaklytuned.mat')); end
if ~exist('iunits_weaklytuned_sort','var'); load(fullfile(figuredir,'iunits_weaklytuned_sort.mat')); end
if ~exist('iunits_remove','var'); load(fullfile(figuredir,'iunits_remove.mat')); end
if ~exist('iunits_posALL','var'); load(fullfile(figuredir,'iunits_posALL.mat')); end
if ~exist('iunits_posALL_sort','var'); load(fullfile(figuredir,'iunits_posALL_sort.mat')); end
if ~exist('iunits_negALL','var'); load(fullfile(figuredir,'iunits_negALL.mat')); end
if ~exist('iunits_negALL_sort','var'); load(fullfile(figuredir,'iunits_negALL_sort.mat')); end
if ~exist('iunits_compass_integratedanglefromRNNoutput','var'); load(fullfile(figuredir,'iunits_compass_integratedanglefromRNNoutput.mat')); end
if ~exist('iunits_shiftpos_integratedanglefromRNNoutput','var'); load(fullfile(figuredir,'iunits_shiftpos_integratedanglefromRNNoutput.mat')); end
if ~exist('iunits_shiftneg_integratedanglefromRNNoutput','var'); load(fullfile(figuredir,'iunits_shiftneg_integratedanglefromRNNoutput.mat')); end
if ~exist('iunits_weaklytuned_integratedanglefromRNNoutput','var'); load(fullfile(figuredir,'iunits_weaklytuned_integratedanglefromRNNoutput.mat')); end
%--------------------------------------------------------------------------
%    plot the firing rates of each unit as a function of integrated angle and angular input
%  integrated angle is computed in two ways: 1) based on integrated inputs (ground truth) and 2) based on RNN outputs (internal estimate of heading direction)
%--------------------------------------------------------------------------
PLOTALLUNITSONTHESAMESCALE = 0;% if 1 plot all units on the same scale
eps = 10^-4;% add epsilon to largest element of anglegrid so when angle equals max(anglegrid) then iangle is not empty 
dangle1 = 1; angle1grid = [0:dangle1:(360-dangle1) 360+eps];% 1 x 361 matrix, discretize the integrated angle
if BOUNDARY.periodic == 0; dangle1 = 1; angle1grid = [BOUNDARY.minangle:dangle1:(BOUNDARY.maxangle-dangle1) BOUNDARY.maxangle+eps]; end% integrated angle cannot go outside BOUNDARY.minangle and BOUNDARY.maxangle
minangularinput = min(min(INtest(1,:,:))) * 180/pi; maxangularinput = max(max(INtest(1,:,:))) * 180/pi; dangle2 = (maxangularinput - minangularinput)/100; angle2grid = [minangularinput:dangle2:(maxangularinput-dangle2) maxangularinput+eps];% 1 x something matrix, discretize the angular input



firingrate_store = -700*ones(numel(angle2grid)-1,numel(angle1grid)-1,numh);
firingrate_normalizeeachrow_store = -700*ones(numel(angle2grid)-1,numel(angle1grid)-1,numh);
firingrate_integratedanglefromRNNoutput_store = -700*ones(numel(angle2grid)-1,numel(angle1grid)-1,numh);
for iunit=1:numh
    firingrate = zeros(numel(angle2grid)-1,numel(angle1grid)-1);
    firingrate_integratedanglefromRNNoutput = zeros(numel(angle2grid)-1,numel(angle1grid)-1);
    numvisits = zeros(numel(angle2grid)-1,numel(angle1grid)-1);% number of times agent visits bin
    numvisits_integratedanglefromRNNoutput = zeros(numel(angle2grid)-1,numel(angle1grid)-1);% number of times agent visits bin
    for itrial=1:numtrials
        RNNoutputangle = -700*ones(1,numTtest);
        RNNoutputangle(:) = atan2(y(1,:,itrial),y(2,:,itrial))*180/pi;% angle between -180 and +180 degrees
        iswitch = find(RNNoutputangle < 0); RNNoutputangle(iswitch) = 360 + RNNoutputangle(iswitch);% angle between 0 and 360 degrees
        for t=1:numTtest
            angle1 = angle_radians(t,itrial)*180/pi;% stored angle in degrees, between 0 and 360
            angle1_integratedanglefromRNNoutput = RNNoutputangle(t);% stored angle in degrees, between 0 and 360
            angle2 = INtest(1,t,itrial)*180/pi;% angular input in degrees
            iangle1 = find(angle1 < angle1grid ,1,'first') - 1;% if xposition < min(angle1grid) iangle1 = 0, if xposition > max(angle1grid) iangle1 is an empty matrix
            iangle1_integratedanglefromRNNoutput = find(angle1_integratedanglefromRNNoutput < angle1grid ,1,'first') - 1;% if xposition < min(angle1grid) iangle1 = 0, if xposition > max(angle1grid) iangle1 is an empty matrix
            iangle2 = find(angle2 < angle2grid ,1,'first') - 1;% if yposition < min(angle2grid) iangle2 = 0, if yposition > max(angle2grid) iangle2 is an empty matrix
            numvisits(iangle2,iangle1) = numvisits(iangle2,iangle1) + 1;
            numvisits_integratedanglefromRNNoutput(iangle2,iangle1_integratedanglefromRNNoutput) = numvisits_integratedanglefromRNNoutput(iangle2,iangle1_integratedanglefromRNNoutput) + 1;
            firingrate(iangle2,iangle1) = firingrate(iangle2,iangle1) + h(iunit,t,itrial);
            firingrate_integratedanglefromRNNoutput(iangle2,iangle1_integratedanglefromRNNoutput) = firingrate_integratedanglefromRNNoutput(iangle2,iangle1_integratedanglefromRNNoutput) + h(iunit,t,itrial);
        end
    end
    firingrate = firingrate ./ numvisits;% if numvisits is 0 firingrate is NaN
    firingrate_integratedanglefromRNNoutput = firingrate_integratedanglefromRNNoutput ./ numvisits_integratedanglefromRNNoutput;% if numvisits is 0 firingrate is NaN
    if (sum(numvisits(:)) ~= numtrials*numTtest); error('missing visits'); end
    if (sum(numvisits_integratedanglefromRNNoutput(:)) ~= numtrials*numTtest); error('missing visits'); end
    
    %  normalize each row (each angular input) to be between 0 and 1
    %  only normalize row if maxrow - minrow > 10^-4, i.e. don't magnify noise fluctations
    firingrate_normalizeeachrow = firingrate;
    for irow=1:size(firingrate,1)
        maxrow = max(firingrate(irow,:));
        minrow = min(firingrate(irow,:));
        if (maxrow - minrow) > eps
            firingrate_normalizeeachrow(irow,:) = (firingrate_normalizeeachrow(irow,:) - minrow) / (maxrow - minrow);% normalize each row to have values between 0 and 1
        end
    end
    
    
    firingrate_store(:,:,iunit) = firingrate;
    firingrate_normalizeeachrow_store(:,:,iunit) = firingrate_normalizeeachrow;
    firingrate_integratedanglefromRNNoutput_store(:,:,iunit) = firingrate_integratedanglefromRNNoutput;
end% for iunit=1:numh

for ifigure=1:5
    %--------------------------------------------------------------------------
    % ifigure==1, firing of all hidden units as a function of integrated angle (true value) and angular input
    % ifigure==2, firing of all hidden units as a function of integrated angle (from RNN output) and angular input
    % ifigure==3, plot the firing rates of each unit as a function of integrated angle (true value) and angular input
    %             normalize each row (each angular input) to be between 0 and 1
    %             (only normalize row if maxrow - minrow > 10^-4, i.e. don't magnify noise fluctations)
    % ifigure==4, firing of all hidden units as a function of integrated angle (true value) and angular input, sort units by posALL,negALL,remove and then anglepreferred_integratedangularinput
    % ifigure==5, firing of all hidden units as a function of integrated angle (true value) and angular input, sort units by compass,shiftpos,shiftneg,weaklytuned,remove and then anglepreferred_integratedangularinput
    %--------------------------------------------------------------------------
    if ifigure==4; desiredorder = [iunits_posALL_sort; iunits_negALL_sort; iunits_remove];                                                firingrate_sortposALLnegALL = -700*ones(numel(angle2grid)-1,numel(angle1grid)-1,numh); end
    if ifigure==5; desiredorder = [iunits_compass_sort; iunits_shiftpos_sort; iunits_shiftneg_sort; iunits_weaklytuned_sort; iunits_remove]; firingrate_sortcompassshiftposshiftneg = -700*ones(numel(angle2grid)-1,numel(angle1grid)-1,numh); end
    if ifigure==4 || ifigure==5; if ~isequal(sort(desiredorder),[1:numh]'); error('missing units, line 2272'); end; end
    
    handle = figure;% firing of all hidden units as a function of integrated angle (true value) and angular input
    clf; hold on; fontsize = 12; set(gcf,'DefaultLineLineWidth',1,'DefaultLineMarkerSize',fontsize,'DefaultAxesLineWidth',1); set(gca,'TickDir','out','FontSize',fontsize,'DefaulttextFontSize',fontsize)
    for iunit=1:numh
        if ifigure==1; firingrate = firingrate_store(:,:,iunit); end
        if ifigure==2; firingrate = firingrate_integratedanglefromRNNoutput_store(:,:,iunit); end
        if ifigure==3; firingrate = firingrate_normalizeeachrow_store(:,:,iunit); end
        if ifigure==4; firingrate = firingrate_store(:,:,iunit); end
        if ifigure==5; firingrate = firingrate_store(:,:,iunit); end
        if ifigure==4 || ifigure==5
            positionofunit = find(desiredorder == iunit);% unit iunit is plotted at position positionofunit
        end
        if ifigure==4; firingrate_sortposALLnegALL(:,:,positionofunit) = firingrate; end% save sorted firing fields
        if ifigure==5; firingrate_sortcompassshiftposshiftneg(:,:,positionofunit) = firingrate; end% save sorted firing fields
        
        numrowsinfigure = ceil(sqrt(numh));
        numcolumnsinfigure = ceil(numh/numrowsinfigure);
        if ifigure<=3; subplot(numrowsinfigure,numcolumnsinfigure,iunit); end
        if ifigure>=4; subplot(numrowsinfigure,numcolumnsinfigure,positionofunit); end
        handleimage = imagesc(angle1grid(1:end-1)+dangle1/2,angle2grid(1:end-1)+dangle2/2,firingrate);% A(1,1) is in the top left corner of the image, if not set(gca,'YDir','reverse')
        set(gca,'YDir','normal')% flip so point of triangle is at top
        set(handleimage,'alphadata',~isnan(firingrate))% if numvisits is 0 firingrate is NaN
        colormap parula;
        if PLOTALLUNITSONTHESAMESCALE
            if isequal(nonlinearity{1},'tanh'); cmin = -1; cmax = 1; caxis([cmin cmax]); end% all values less than cmin will have the same color as cmin, namely blue
            if isequal(nonlinearity{1},'retanh'); cmin = 0; cmax = 1; caxis([cmin cmax]); end% all values less than cmin will have the same color as cmin, namely blue
            if isequal(nonlinearity{1},'logistic'); cmin = 0; cmax = 1; caxis([cmin cmax]); end% all values less than cmin will have the same color as cmin, namely blue
            if isequal(nonlinearity{1},'ReLU') && ifigure==1; cmin = 0; cmax = max(firingrate_store(:)); caxis([cmin cmax]); end% all values less than cmin will have the same color as cmin, namely blue
            if isequal(nonlinearity{1},'ReLU') && ifigure==2; cmin = 0; cmax = max(firingrate_integratedanglefromRNNoutput_store(:)); caxis([cmin cmax]); end% all values less than cmin will have the same color as cmin, namely blue
            if isequal(nonlinearity{1},'ReLU') && ifigure==3; cmin = 0; cmax = max(firingrate_normalizeeachrow_store(:)); caxis([cmin cmax]); end% all values less than cmin will have the same color as cmin, namely blue
        end
        % if unit is "dead" color it blue
        if isequal(nonlinearity{1},'retanh') && max(firingrate(:))==0; cmin = 0; cmax = 1; caxis([cmin cmax]); end% all values less than cmin will have the same color as cmin, namely blue
        
        set(gca,'xtick',[],'ytick',[]);
        %set(gca,'Visible','off')% uncomment for final figure
        %title(sprintf('%.2g',tuningspatial_lifetimesparseness(iunit))); box off;% comment for final figure
        if ifigure==1 || ifigure==3% || ifigure==5
            if ismember(iunit,iunits_compass); title('compass','fontsize',2,'fontweight','normal'); end% ax = gca; ax.FontSize = 1;
            if ismember(iunit,iunits_shiftpos); title('shift pos','fontsize',2,'fontweight','normal'); end% ax = gca; ax.FontSize = 1;
            if ismember(iunit,iunits_shiftneg); title('shift neg','fontsize',2,'fontweight','normal'); end% ax = gca; ax.FontSize = 1;
            if ismember(iunit,iunits_weaklytuned); title('weakly tuned','fontsize',2,'fontweight','normal'); end% ax = gca; ax.FontSize = 1;
        end
        if ifigure==2
            if ismember(iunit,iunits_compass_integratedanglefromRNNoutput); title('compass','fontsize',2,'fontweight','normal'); end% ax = gca; ax.FontSize = 1;
            if ismember(iunit,iunits_shiftpos_integratedanglefromRNNoutput); title('shift pos','fontsize',2,'fontweight','normal'); end% ax = gca; ax.FontSize = 1;
            if ismember(iunit,iunits_shiftneg_integratedanglefromRNNoutput); title('shift neg','fontsize',2,'fontweight','normal'); end% ax = gca; ax.FontSize = 1;
            if ismember(iunit,iunits_weaklytuned_integratedanglefromRNNoutput); title('weakly tuned','fontsize',2,'fontweight','normal'); end% ax = gca; ax.FontSize = 1;
        end
        if ifigure==4
            if ismember(iunit,iunits_posALL); title('pos','fontsize',2,'fontweight','normal'); end% ax = gca; ax.FontSize = 1;
            if ismember(iunit,iunits_negALL); title('neg','fontsize',2,'fontweight','normal'); end% ax = gca; ax.FontSize = 1;
        end
        %set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
        %axis tight; axis equal
        axis([min(angle1grid(1:end-1)+dangle1/2)  max(angle1grid(1:end-1)+dangle1/2)  min(angle2grid(1:end-1)+dangle2/2)  max(angle2grid(1:end-1)+dangle2/2)])
        drawnow
        %pause
    end% for iunit=1:numh
    if ifigure==4; save(fullfile(figuredir,'firingrate_sortposALLnegALL.mat'),'firingrate_sortposALLnegALL'); end
    if ifigure==5; save(fullfile(figuredir,'firingrate_sortringshiftposshiftneg.mat'),'firingrate_sortcompassshiftposshiftneg'); end
    
    if ifigure==1 || ifigure==3 || ifigure==4 || ifigure==5; [ax1,h1]=suplabel(sprintf('Integrated angular input (%.3g to %.3g degrees)',min(angle1grid),max(angle1grid)),'x'); end
    if ifigure==2;               [ax1,h1]=suplabel(sprintf('Integrated angular input from RNN output (%.3g to %.3g degrees)',min(angle1grid),max(angle1grid)),'x'); end
    [ax2,h2]=suplabel(sprintf('Angular input (%g to %g degrees)',min(angle2grid),max(angle2grid)),'y');
    if PLOTALLUNITSONTHESAMESCALE==0 && (ifigure==1 || ifigure==2 || ifigure==4 || ifigure==5); [a3, h3] = suplabel({[sprintf('Different scale for each unit, Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)]},'t'); end
    if PLOTALLUNITSONTHESAMESCALE==0 && ifigure==3; [a3, h3] = suplabel({[sprintf('Different scale for each unit (normalize each row), Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)]},'t'); end
    if PLOTALLUNITSONTHESAMESCALE==1
        if isequal(nonlinearity{1},'tanh'); [a3, h3] = suplabel({[sprintf('Same scale for each unit, Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)]},'t'); end
        if isequal(nonlinearity{1},'retanh'); [a3, h3] = suplabel({[sprintf('Scale for each unit is from 0 to 1, Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)]},'t'); end
        if isequal(nonlinearity{1},'logistic'); [a3, h3] = suplabel({[sprintf('Scale for each unit is from 0 to 1, Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)]},'t'); end
        if isequal(nonlinearity{1},'ReLU'); [a3, h3] = suplabel({[sprintf('Scale for each unit is from 0 to max(firingrateofallunits), Training epoch %g, %g trials, %g timesteps in simulation',epoch,numtrials,numTtest)];[sprintf('normalized error overall = %.5g%%',normalizederror)]},'t'); end
    end
    set(h1,'fontsize',fontsize); set(h2,'fontsize',fontsize); set(h3,'fontsize',fontsize,'fontweight','normal')
    %set(gca,'FontSize',fontsize,'fontWeight','normal'); set(findall(gcf,'type','text'),'FontSize',fontsize,'fontWeight','normal')
    if ifigure==1
        if PLOTALLUNITSONTHESAMESCALE==0; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinputandintegratedangle_epoch%g_numT%g',figuredir,epoch,numTtest)); end
        if PLOTALLUNITSONTHESAMESCALE==1; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinputandintegratedangle_epoch%g_numT%g_samecolorscaleforallunits',figuredir,epoch,numTtest)); end
    end
    if ifigure==2
        if PLOTALLUNITSONTHESAMESCALE==0; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinputandintegratedangle_epoch%g_numT%g_integratedanglefromRNNoutput',figuredir,epoch,numTtest)); end
        if PLOTALLUNITSONTHESAMESCALE==1; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinputandintegratedangle_epoch%g_numT%g_samecolorscaleforallunits_integratedanglefromRNNoutput',figuredir,epoch,numTtest)); end
    end
    if ifigure==3
        if PLOTALLUNITSONTHESAMESCALE==0; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinputandintegratedangle_epoch%g_numT%g_normalizeeachrow',figuredir,epoch,numTtest)); end
        if PLOTALLUNITSONTHESAMESCALE==1; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinputandintegratedangle_epoch%g_numT%g_samecolorscaleforallunits_normalizeeachrow',figuredir,epoch,numTtest)); end
    end
    if ifigure==4
        if PLOTALLUNITSONTHESAMESCALE==0; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinputandintegratedangle_epoch%g_numT%g_sortposALLnegALL',figuredir,epoch,numTtest)); end
        if PLOTALLUNITSONTHESAMESCALE==1; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinputandintegratedangle_epoch%g_numT%g_samecolorscaleforallunits_sortposALLnegALL',figuredir,epoch,numTtest)); end
    end
    if ifigure==5
        if PLOTALLUNITSONTHESAMESCALE==0; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinputandintegratedangle_epoch%g_numT%g_sortcompassshiftposshiftneg',figuredir,epoch,numTtest)); end
        if PLOTALLUNITSONTHESAMESCALE==1; print(handle, '-dpdf', sprintf('%s/allunits_meanhVSangularinputandintegratedangle_epoch%g_numT%g_samecolorscaleforallunits_sortcompassshiftposshiftneg',figuredir,epoch,numTtest)); end
    end
end% for ifigure=1:5








