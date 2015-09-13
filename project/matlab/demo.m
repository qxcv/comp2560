% % Copyright (C) 2014 LEAR, Inria Grenoble, France
% 
% Permission is hereby granted, free of charge, to any person obtaining
% a copy of this software and associated documentation files (the
% "Software"), to deal in the Software without restriction, including
% without limitation the rights to use, copy, modify, merge, publish,
% distribute, sublicense, and/or sell copies of the Software, and to
% permit persons to whom the Software is furnished to do so, subject to
% the following conditions:
% 
% The above copyright notice and this permission notice shall be
% included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
% NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
% LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
% OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
% WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

%%
% If you use this code, please cite: 
% @INPROCEEDINGS{cherian14, 
% author={Cherian, A. and Mairal, J. and Alahari, K. and Schmid, C.}, 
% booktitle={IEEE Conference on Computer Vision and Pattern Recognition}, 
% title={Mixing Body-Part Sequences for Human Pose Estimation}, 
% year={2014}
% }
% 
% demo code main file to run.
% for bugs contact anoop.cherian@inria.fr
%
warning 'off';
startup(); % set some paths

% configure. See the function for details. You need to set the cache and
% the sequence paths in the config.
config = set_algo_parameters();     

% load the bodypart model learned using Yang and Ramanan framework.
load ./data/FLIC_model.mat;% the 13part FLIC human pose model.

% get the dataset and gt annotations
piw_data = get_piw_data('piw', config.data_path);

% process each sequence, here we show only for seq15 available in the
% dataset folder. 
detected_pose_type = struct('seq', {}, 'filename', {}, 'frame', {}, 'bestpose',{});
detected_pose_seqs = repmat(detected_pose_type, [1,1,1]);
seqs = dir(config.data_path); seqs = seqs(3:end);
all_detections = []; gt_all = []; mov = 1;

% for every sequence in the selected_seqs folder (with the dataset)
for s=1:length(seqs)
    fprintf('working on sequence %s\n', seqs(s).name);

    % read the frames and store the respective groundtruth annotations.
    seq_dir = [config.data_path seqs(s).name '/'];    
    frames = dir([seq_dir '/*.png']);    
    gt = get_groundtruth_for_seq(frames, piw_data);% extract gt annotations for the frames in seq
    gt_all = [gt_all, gt]; % used for full evaluation.

    % now we are ready to compute the part sequences and recombination!               
    try        
        load([config.data_store_path 'detected_poses_' seqs(s).name], 'detected_poses'); 
    catch
        detected_poses = EstimatePosesInVideo(seq_dir, model, 1, config);   
        save([config.data_store_path '/detected_poses_' seqs(s).name], 'detected_poses');
    end
    
    % fill in some structure for the complete evaluation
    for i=1:length(frames)              
        detected_pose_seqs(mov, s, i).seq = seqs(s).name;
        detected_pose_seqs(mov, s, i).filename = frames(i).name;            
        detected_pose_seqs(mov, s, i).frame = get_framenum(frames(i).name);        

        % if we could find a pose sequence path
        if ~isempty(detected_poses)            
            detected_pose_seqs(mov, s, i).bestpose = detected_poses(i,:);            
        else
            detected_pose_seqs(mov, s, i).bestpose = [];
        end
    end        
    show_pose_sequence(seq_dir, frames, detected_poses);
end
%
% evaluate the sequences for pixel error
pix_error = evaluate_pose_seqs(detected_pose_seqs, gt_all, config.pix_thresh);  
fprintf('pixel error @ %d for Shol=%0.4f Elbow=%0.4f Wrist=%0.4f \n', config.pix_thresh, ...
                    max(pix_error([2,7])), max(pix_error([4,9])), max(pix_error([6,11])));                
