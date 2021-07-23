clc
clear all
close all

width = 0.15;
fontsize = 14;
% protonet results for p = 1, n = 10. 1st item: C = 2, K = 1; 2nd item: C = 2, K =
% 5; 3rd item: C = 5, K = 1; 4th item: C = 5, K = 5; 5th item: C = 10, K =
% 1; 6th item: C = 10, K = 5; 7th item: C = 10, K = 10
protonet_p1 = 100*[0.7313661633500341, 0.7216052646050743, 0.709344626421641, ...
    0.709344626421641, 0.7289736335075017, 0.7273792702003725, 0.7250145211827318];
protonet_pos_err_p1 = 100*[0.06958770751665745, 0.07480030320486489, 0.08932576878847527,...
    0.08932576878847527, 0.0674421339154109, 0.07830704880480445, 0.05943034481718991];
protonet_neg_err_p1  = -protonet_pos_err_p1;

step = 5.5;
x1 = 1: step :length(protonet_p1)*step-3;

figure('DefaultAxesFontSize',fontsize);
bar(x1, protonet_p1, width)        
hold on

% protonet results for p = 5, n = 10.
protonet_p5 = 100*[0.8007797113739553, 0.791952229985763, 0.7748523879935596,...
    0.7541069275528921, 0.7878775998607037, 0.7946936647425777, 0.7935081229902037];
protonet_pos_err_p5 = 100*[0.056745150368195794,0.06256920351059289,0.09230308729029153,...
    0.07564008570976402, 0.05712458769308526, 0.07007064774816178, 0.050143887383148525];
protonet_neg_err_p5  = -protonet_pos_err_p5;
x2 = x1 + 1;

bar(x2, protonet_p5, width)        
hold on

% relation results for p = 1, n = 10.
relation_p1 = 100*[0.7755241799399051, 0.7755241799399051, 0.889829151776634, 0.6666631660433195,...
    0.8980601743442378, 0.7108430580244031, 0.5389955728381808];
relation_pos_err_p1 = 100*[0.06909424284532331, 0.04677794890217549, 0.061127963096542864,...
    0.04711225511641137, 0.053929538964086386, 0.05991458158077961, 0.015176479236748686];
relation_neg_err_p1  = -relation_pos_err_p1;
x3 = x1 + 2;

bar(x3, relation_p1, width)        
hold on

% relation results for p = 5, n = 10.
relation_p5 = 100*[0.596667006643279, 0.8137760315972479, 0.6984828121747196, ...
    0.9451014510975475, 0.6912486051677477, 0.9459941156074188, 0.8456392393872497];
relation_pos_err_p5 = 100*[0.042636893116736574, 0.05658867045694816, 0.05061710277538254,...
    0.037304523815161424, 0.05267813281857105, 0.03655259479177812, 0.061225120883054346];
relation_neg_err_p5  = -relation_pos_err_p5;
x4 = x1 + 3;

bar(x4, relation_p5, width)        
hold on

xlabel("(C, K)",'FontSize', fontsize);
ylabel("AUPRC (%)",'FontSize', fontsize);
xticks(2.5: step : (length(protonet_p1))*step-2.5);
xticklabels(["(2,1)", "(2,5)", "(5,1)", "(5,5)", "(10,1)", "(10,5)", "(10,10)"]);
legend(["Prototypical, p=1", "Prototypical, p=5", "Relation, p=1", "Relation, p=5"], 'location', 'bestoutside','FontSize', fontsize);
ylim([0, 100]);

er = errorbar(x1,protonet_p1, protonet_pos_err_p1, protonet_neg_err_p1, 'HandleVisibility','off');    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  

hold on 

er = errorbar(x2,protonet_p5, protonet_pos_err_p5, protonet_neg_err_p5, 'HandleVisibility','off');    
er.Color = [0 0 0];                            
er.LineStyle = 'none'; 
hold on 

er = errorbar(x3,relation_p1, relation_pos_err_p1, relation_neg_err_p1, 'HandleVisibility','off');    
er.Color = [0 0 0];                            
er.LineStyle = 'none'; 
hold on 

er = errorbar(x4,relation_p5, relation_pos_err_p5, relation_neg_err_p5, 'HandleVisibility','off');    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  

hold off