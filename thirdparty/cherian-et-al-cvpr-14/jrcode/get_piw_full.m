function test_seqs = get_piw_full(cache_dir, trans_spec)
%GET_PIW_FULL Get Poses in the Wild dataset

% List of weird gotchas I've found in PIW:
%  - Joints which aren't visible (even if their location can be inferred
%    despite occlusion) are set to [-1, -1]
%  - Joint annotations drift over time, sometimes quite severely.
%  - There are two frame discontinuities which I could find within the same
%    sequence:
%  - One in seq20 between The-Terminal-3768-3785-002{35,79}.png (actor
%    moves by large amount and swivels chair significantly between frames)
%  - Another in seq73 between The-Terminal-6452-6463-00{089,123}.png
%    (camera jerkily zooms out between frames, after smoothly zooming in
%    frames before).

% Also, PIW skeleton is:
% 1, 8: Neck (more like chin), torso
% 2, 3, 4: Left shoulder, elbw, wrist
% 5, 6, 7: Right shoulder, elbow, wrist
piw_url = 'https://lear.inrialpes.fr/research/posesinthewild/dataset/poses_in_the_wild_public.tar.gz';
cache_path = fullfile(cache_dir, 'poses_in_the_wild_public.tar.gz');
dest_path = fullfile(cache_dir, 'poses_in_the_wild_public');

mkdir_p(cache_dir);

if ~exist(dest_path, 'dir')
    if ~exist(cache_path, 'file')
        fprintf('Downloading PIW from %s\n', piw_url);
        websave(cache_path, piw_url);
    end
    fprintf('Extracting PIW data to %s\n', dest_path);
    % Extract  to ds_dir and it will show up at dest_path
    untar(cache_path, cache_dir);
end

% Seqs are already split for us, so this will be really easy.
piw_data = parload(fullfile(dest_path, 'poses_in_the_wild_data.mat'), ...
    'piw_data');
num_data = length(piw_data);
empty = cell([1 num_data]);
data = struct('image_path', empty, 'orig_joint_locs', empty, ...
    'joint_locs', empty, 'orig_visible', empty, 'visible', empty, ...
    'orig_seq', empty, 'frame_no', empty);

% We'll trigger a change to a new sequence number every time the frame
% threshold is violated or the annotated sequence number changes.
seqs = {};
prev_frame_num = [];
prev_seq_num = [];
% If we skip over >= skip_thresh frames with a labelled sequence then we
% assume there's a problem.
skip_thresh = 5;

for data_idx=1:num_data
    orig_datum = piw_data(data_idx);
    datum.image_path = fullfile(dest_path, orig_datum.im);
    datum.orig_joint_locs = orig_datum.point;
    datum.joint_locs = skeltrans(orig_datum.point, trans_spec);
    datum.orig_visible = trans_visible(orig_datum.visible, trans_spec);
    datum.visible = trans_visible(orig_datum.visible, trans_spec);
    [seq_num, frame_num] = get_seq_frame(datum.image_path);
    datum.orig_seq = seq_num;
    datum.frame_no = data_idx;
    datum.joint_locs(~datum.visible, :) = nan;
    data(data_idx) = datum;
    
    % Now handle sequences
    if data_idx > 1
        have_seq_skip = ~isempty(prev_seq_num) && seq_num ~= prev_seq_num;
        frame_diff = frame_num - prev_frame_num;
        have_frame_skip = ~isempty(prev_frame_num) && ...
            (frame_diff >= skip_thresh || frame_diff <= 0);
    end
    if data_idx == 1 || have_seq_skip || have_frame_skip
        % Add in a new sequence if we see a skip
        seqs{length(seqs)+1} = []; %#ok<AGROW>
    end
    seqs{end} = [seqs{end} data_idx];
    
    prev_seq_num = seq_num;
    prev_frame_num = frame_num;
end

check_vis(data);

test_seqs.data = data;
test_seqs.seqs = seqs;
test_seqs.name = 'piw_test_seqs';
end

function [seq_num, frame_num] = get_seq_frame(filename)
[~, tokens, ~] = regexp(filename, '/selected_seqs/seq(\d+)/.+-(\d+)\.png$', ...
    'match', 'tokens');
assert(length(tokens) == 1 && length(tokens{1}) == 2);
seq_str = tokens{1}{1};
seq_num = str2double(seq_str);
frame_str = tokens{1}{2};
frame_num = str2double(frame_str);
end

function check_vis(data)
% Check that a joint is marked invisible if it has a [-1, -1] annotation
% (but not converse, which fails to hold for some reason)
for i=1:length(data)
    jl = data(i).joint_locs;
    invalid = any(abs(jl + 1) < 1e-5, 2);
    visible = data(i).visible;
    % Read this is as "invalid implies invisible"
    assert(all(~invalid | ~visible));
end
end

function rv = trans_visible(visible, trans_spec)
% Much like skeltrans, but transforms the indicator array of visible joints
% so that it's valid for the new skeleton model (which might include more
% joints than the old one, because of composites)
rv = true([length(trans_spec) 1]);
for i=1:length(trans_spec)
    % A joint in the new skeleton is visible iff all of the joints which
    % it's composed of in the old skeleton are visible
    rv(i) = all(visible(trans_spec(i).indices));
end
end
