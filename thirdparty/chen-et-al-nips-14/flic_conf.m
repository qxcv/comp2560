function conf = flic_conf(conf)
% dataset
conf.dataset = 'flic';
conf.step = 6;
% conf.K = 11;
conf.K = 13;
conf.NEG_N = 100;

conf.test_with_detection = true;  % constraint neck point to be inside a box
conf.constrainted_pids = 2;  % neck point
% for full body
% 18 part
conf.pa = [0 1 2 3 4 5 6 3 8 9 2 11 12 13 14 11 16 17];
d_step = 5;
conf.degree = [-35:d_step:-d_step,d_step:d_step:35];

% ---------------- CNN files ------------------
conf.cnn.cnn_deploy_conv_file = './external/my_models/flic/flic_deploy_conv.prototxt';
conf.cnn.cnn_conv_model_file = './cache/flic/fully_conv_net_by_net_surgery.caffemodel';
conf.cnn.cnn_deploy_file = './external/my_models/flic/flic_deploy.prototxt';
conf.cnn.cnn_model_file = './cache/flic/flic_iter_60000.caffemodel';

% ----- evaluation functions -----
conf.reference_joints_pair = [6, 7];     % right shoulder and left hip (from observer's perspective)
conf.symmetry_joint_id = [2,1,7,8,9,10,3,4,5,6];
conf.show_joint_ids = find(conf.symmetry_joint_id >= 1:10); 
conf.joint_name = {'Head', 'Shou', 'Elbo', 'Wris', 'Hip'};

conf.box2det = @flic_box2det;
conf.joint2stick = @flic_joint2stick;
conf.symmetry_part_id = [1,2,5,6,3,4];
conf.show_part_ids = find(conf.symmetry_part_id >= 1:numel(conf.symmetry_part_id));
conf.part_name = {'Head', 'Torso', 'U.arms', 'L.arms'};

% -------------
conf.impyra_fun = @impyra;

