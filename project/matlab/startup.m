% set some paths!
if ~exist('first_time', 'var')    
    % addpaths!
    addpath(genpath('./YR/'));
    addpath ./utils/;
    addpath ./mex/;
    addpath ./eval/;
    addpath(genpath('./flow/'));    
    addpath ./detect/;
    
    LDOF_startup; % compile LDOF c files    
    YR_compile; % compile YR c files.
    CY_startup;
    CY_compile;
    mex ./mex/ksp.cpp -outdir ./mex/;                      
    mex ./mex/mymax.cpp -outdir ./mex;
        
    first_time = 1;
end
