function test_seqs = get_mpii_cooking(dest_dir, cache_dir, trans_spec)
%GET_MPII_COOKING Fetches continuous pose estimation data from MPII
% train_dataset and val_dataset both come from MPII Cooking's continuous
% pose dataset, with val_dataset scraped from a couple of scenes at the
% end. test_dataset comes from the training set of MPII Cooking's "pose
% challenge". The training set of the pose challenge happens to be
% continuous, but it is *not* the same as the continuous pose dataset
% (which I believe was derived from the same source video as the pose
% challenge *after* Cooking Activities was released).
MPII_POSE_URL = 'http://datasets.d2.mpi-inf.mpg.de/MPIICookingActivities/poseChallenge-1.1.zip';
POSE_DEST_PATH = fullfile(dest_dir, 'mpii-cooking-pose-challenge');
POSE_CACHE_PATH = fullfile(cache_dir, 'poseChallenge-1.1.zip');
VAL_FRAME_SKIP = 0;
DUMP_THRESH = 50; % Set in main repo (joint-regressor.git)

data_path = fullfile(cache_dir, 'mpii_data_piwcompat.mat');
if exist(data_path, 'file')
    fprintf('Found existing data, so I''ll just use that\n');
    load(data_path, 'test_seqs');
    return
else
    fprintf('Need to regenerate all data :(\n');
end

% Next we grab the smaller original pose estimation dataset, which has a
% continuous (but low-FPS) training set but discontinuous testing set.
% We'll use the training set from that as our validation set.
if ~exist(POSE_DEST_PATH, 'dir')
    if ~exist(POSE_CACHE_PATH, 'file')
        fprintf('Downloading MPII pose challenge from %s\n', MPII_POSE_URL);
        websave(POSE_CACHE_PATH, MPII_POSE_URL);
    end
    fprintf('Extracting basic MPII pose challenge data to %s\n', POSE_DEST_PATH);
    unzip(POSE_CACHE_PATH, POSE_DEST_PATH);
end

test_data = load_files_basic(POSE_DEST_PATH, trans_spec);
test_data = split_mpii_scenes(test_data, 0.1);
test_pairs = find_pairs(test_data, VAL_FRAME_SKIP, DUMP_THRESH);
test_dataset = unify_dataset(test_data, test_pairs, 'test_dataset_mpii_base');
% Need 10% frames per sequence
all_test_seqs = pairs2seqs(test_dataset, 10);

fprintf('Test set has %i seqs and %i frames\n', ...
    length(all_test_seqs), sum(cellfun(@length, all_test_seqs)));
test_seqs = make_test_set(test_dataset, all_test_seqs);

% Cache
save(data_path, 'test_seqs');
end

function basic_data = load_files_basic(dest_path, trans_spec)
pose_dir = fullfile(dest_path, 'data', 'train_data', 'gt_poses');
pose_fns = dir(pose_dir);
pose_fns = pose_fns(3:end);
basic_data = struct(); % Silences Matlab warnings about growing arrays
for fn_idx=1:length(pose_fns)
    data_fn = pose_fns(fn_idx).name;
    frame_no = parse_basic_fn(data_fn);
    basic_data(fn_idx).frame_no = frame_no;
    file_name = sprintf('img_%06i.jpg', frame_no);
    basic_data(fn_idx).image_path = fullfile(dest_path, 'data', 'train_data', 'images', file_name);
    loaded = load(fullfile(pose_dir, data_fn), 'pose');
    basic_data(fn_idx).orig_joint_locs = loaded.pose;
    basic_data(fn_idx).joint_locs = skeltrans(loaded.pose, trans_spec);
    basic_data(fn_idx).is_val = true;
end
basic_data = sort_by_frame(basic_data);
end

function sorted = sort_by_frame(data)
[~, sorted_indices] = sort([data.frame_no]);
sorted = data(sorted_indices);
end

function index = parse_basic_fn(fn)
tokens = regexp(fn, '[^\d]*(\d+)', 'tokens');
assert(length(tokens) >= 1);
assert(length(tokens{1}) == 1);
index = str2double(tokens{1}{1});
end

function pairs = find_pairs(data, frame_skip, dump_thresh)
% Find pairs with frame_skip frames between them
frame_nums = [data.frame_no];
scene_nums = [data.scene_num];
fst_inds = 1:(length(data)-frame_skip-1);
snd_inds = fst_inds + frame_skip + 1;
good_inds = (frame_nums(snd_inds) - frame_nums(fst_inds) == frame_skip + 1) ...
    & (scene_nums(fst_inds) == scene_nums(snd_inds));
dropped = 0;
for i=find(good_inds)
    fst = data(fst_inds(i));
    snd = data(snd_inds(i));
    if mean_dists(fst.joint_locs, snd.joint_locs) > dump_thresh
        fprintf('Dropping %s->%s (%i->%i)\n', ...
            fst.image_path, snd.image_path, fst_inds(i), snd_inds(i));
        good_inds(i) = false;
        dropped = dropped + 1;
    end
end
if dropped
    fprintf('Dropped %i pairs due to threshold %f\n', dropped, dump_thresh);
end
pairs = cat(2, fst_inds(good_inds)', snd_inds(good_inds)');
end

% From the README:
%
% The ground-truth poses for the dataset are provided as .mat files in 'gt_poses' directory.
% 
% The pose files are given as, pose_<TrackIndex>_<Index>.mat
% and the images are given as,  img_<TrackIndex>_<Index>.mat
% 
% <TrackIndex> is the index of the continuous image sequence (activity track)
% <Index> is just the image index in this evaluation set.
%
% Each pose file contains the location of 10 parts (torso and head each consists of two points),
%
%      pose(1,:) -> torso upper point
%      pose(2,:) -> torso lower point
%      pose(3,:) -> right shoulder
%      pose(4,:) -> left shoulder
%      pose(5,:) -> right elbow
%      pose(6,:) -> left elbow
%      pose(7,:) -> right wrist
%      pose(8,:) -> left wrist
%      pose(9,:) -> right hand
%      pose(10,:)-> left hand
%      pose(11,:)-> head upper point
%      pose(12,:)-> head lower point
