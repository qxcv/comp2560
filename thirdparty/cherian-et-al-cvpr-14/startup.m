% set some paths!
if ~exist('first_time', 'var')    
    % addpaths!
    addpath(genpath('./YR/'));
    addpath ./utils/;
    addpath ./mex/;
    addpath ./eval/;
    addpath(genpath('./flow/'));    
    addpath ./detect/;
    addpath ./jrcode/;
    
	LDOF_startup; % compile LDOF c files    
    YR_compile; % compile YR c files.
    mex ./mex/ksp.cpp -outdir ./mex/;                      
    mex ./mex/mymax.cpp -outdir ./mex;
        
    first_time = 1;
end
