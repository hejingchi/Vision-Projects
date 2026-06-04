function accuracy = showAccuracy(results)
correct = 0;
correctA=0; totalA=0;
correctC=0; totalC=0;
correctF=0; totalF=0;
correctV=0; totalV=0;


for i = 1 : size(results,1)
    trueLabel = results{i,1};
    predicted = results{i,2};
    if strcmp(trueLabel, predicted), correct = correct+1; end
    switch trueLabel
        case 'A', totalA=totalA+1; if strcmp(predicted,'A'), correctA=correctA+1; end
        case 'C', totalC=totalC+1; if strcmp(predicted,'C'), correctC=correctC+1; end
        case 'F', totalF=totalF+1; if strcmp(predicted,'F'), correctF=correctF+1; end
        case 'V', totalV=totalV+1; if strcmp(predicted,'V'), correctV=correctV+1; end
    end
end

%fprintf('总准确率: %.1f%% (%d/%d)\n', correct/size(results,1)*100, correct, size(results,1));
%if totalA>0, fprintf('  A: %.1f%% (%d/%d)\n', correctA/totalA*100, correctA, totalA); end
%if totalC>0, fprintf('  C: %.1f%% (%d/%d)\n', correctC/totalC*100, correctC, totalC); end
%if totalF>0, fprintf('  F: %.1f%% (%d/%d)\n', correctF/totalF*100, correctF, totalF); end
%if totalV>0, fprintf('  V: %.1f%% (%d/%d)\n', correctV/totalV*100, correctV, totalV); end

accuracy = {
    'A', correctA, totalA, correctA/totalA*100;
    'C', correctC, totalC, correctC/totalC*100;
    'F', correctF, totalF, correctF/totalF*100;
    'V', correctV, totalV, correctV/totalV*100;
    '总', correct/size(results,1)*100, correct, size(results,1)
};