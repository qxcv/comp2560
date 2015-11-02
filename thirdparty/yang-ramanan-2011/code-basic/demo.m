function demo(stats_path)

addpath visualization;
addpath eval;
addpath utils;
if isunix()
  addpath mex_unix;
elseif ispc()
  addpath mex_pc;
end

compile;

if ~exist('cache', 'dir')
    mkdir cache
end

% load and display model
load('BUFFY_model');

% imlist = dir('images/*.jpg');
% for i = 1:length(imlist)
%     % load and display image
%     im = imread(['images/' imlist(i).name]);
%     clf; imagesc(im); axis image; axis off; drawnow;
% 
%     % call detect function
%     tic;
%     boxes = detect_fast(im, model);
%     dettime = toc; % record cpu time
%     % Get best-scoring box
%     [~,I] = sort(boxes(:,end),'descend');
%     boxes = boxes(I(1:1),:);
%     colorset = {'g','g','y','m','m','m','m','y','y','y','r','r','r','r','y','c','c','c'};
%     showboxes(im, boxes(1,:),colorset);
%     fprintf('detection took %.1f seconds\n',dettime);
%     disp('press any key to continue');
%     pause;
% end
% 
% disp('done');

% get the dataset and gt annotations
cache_path = 'cache/';
seq_path = 'piw/selected_seqs/';
piw_data = get_piw_data('piw', seq_path);

% process each sequence, here we show only for seq15 available in the
% dataset folder.
detected_pose_type = struct('seq', {}, 'filename', {}, 'frame', {}, 'bestpose',{});
detected_pose_seqs = repmat(detected_pose_type, [1,1,1]);
seqs = dir(seq_path); seqs = seqs(3:end);
all_detections = []; gt_all = []; mov = 1;

for s=1:length(seqs)
    fprintf('working on sequence %s\n', seqs(s).name);

    % read the frames and store the respective groundtruth annotations.
    seq_dir = [seq_path seqs(s).name '/'];
    frames = dir([seq_dir '/*.png']);
    gt = get_groundtruth_for_seq(frames, piw_data);
    gt_all = [gt_all, gt];

    try
        load([cache_path 'detected_poses_' seqs(s).name], 'detected_poses');
    catch
        for frame_idx=1:length(frames)
            fprintf('frame %i/%i\n', frame_idx, length(frames));
            img_path = fullfile(seq_dir, frames(frame_idx).name);
            im = imread(img_path);
            box = detect_fast(im, model);
            % Get best-scoring box
            [~,I] = sort(box(:,end),'descend');
            box = box(I(1:1),:);
            assert(size(box, 1) == 1);
            if ~exist('detected_poses', 'var')
                detected_poses = box;
            else
                detected_poses = cat(1, detected_poses, box);
            end
        end
        
        save([cache_path 'detected_poses_' seqs(s).name], 'detected_poses');
    end

    % fill in some structure for the complete evaluation
    for i=1:length(frames)
        detected_pose_seqs(mov, s, i).seq = seqs(s).name;
        detected_pose_seqs(mov, s, i).filename = frames(i).name;
        detected_pose_seqs(mov, s, i).frame = get_framenum(frames(i).name);

        if ~isempty(detected_poses)
            detected_pose_seqs(mov, s, i).bestpose = detected_poses(i,:);
        else
            detected_pose_seqs(mov, s, i).bestpose = [];
        end
    end
end

% evaluate the sequences for pixel error
fprintf('evaluating pixel error\n')
thresholds = 0:2.5:40;
pix_error = zeros(length(thresholds), 11);
for i=1:length(thresholds)
    thresh = thresholds(i);
    pix_error(i, :) = evaluate_pose_seqs(detected_pose_seqs, gt_all, thresh);
end

out_tab = table(...
    thresholds', ...
    max(pix_error(:, [2, 7]), [], 2), ...
    max(pix_error(:, [4, 9]), [], 2), ...
    max(pix_error(:, [6, 11]), [], 2), ...
    'VariableNames', {'Threshold', 'Shoulder', 'Elbow', 'Wrist'});

if nargin == 0
    fprintf('No output file specified, writing stats to console\n');
    disp(out_tab);
else
    fprintf(['Writing stats to ' stats_path '\n']);
    writetable(out_tab, stats_path);
end
end