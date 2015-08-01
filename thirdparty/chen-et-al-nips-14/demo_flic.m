startup;
clear mex; 
global GLOBAL_OVERRIDER;
GLOBAL_OVERRIDER = @flic_conf;
conf = global_conf();
cachedir = conf.cachedir;
pa = conf.pa;
p_no = length(pa);
note = [conf.note];
diary([cachedir note '_log_' datestr(now,'mm-dd-yy') '.txt']);

% -------------------------------------------------------------------------
% read data 
% -------------------------------------------------------------------------
[pos_train, pos_val, pos_test, neg_train, neg_val, tsize] = FLIC_data();
% -------------------------------------------------------------------------
% train dcnn
% -------------------------------------------------------------------------
caffe_solver_file = 'external/my_models/flic/flic_solver.prototxt';
train_dcnn(pos_train, pos_val, neg_train, tsize, caffe_solver_file);
% -------------------------------------------------------------------------
% train graphical model
% -------------------------------------------------------------------------
model = train_model(note, pos_val, neg_val, tsize);
% -------------------------------------------------------------------------
% testing
% -------------------------------------------------------------------------
boxes = test_model([note,'_FLIC'], model, pos_test);
% -------------------------------------------------------------------------
% evaluation
% -------------------------------------------------------------------------
eval_method = {'strict_pcp', 'pdj'};
fprintf('============= On test =============\n');
ests = conf.box2det(boxes, p_no);
% generate part stick from joints locations
for ii = 1:numel(ests)
  ests(ii).sticks = conf.joint2stick(ests(ii).joints);
  pos_test(ii).sticks = conf.joint2stick(pos_test(ii).joints);
end
show_eval(pos_test, ests, conf, eval_method);
diary off;
clear mex;
