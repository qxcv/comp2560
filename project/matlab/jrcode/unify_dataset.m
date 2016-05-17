function unified = unify_dataset(frame_data, pairs, ds_name)
%UNIFY_DATASET Make unified struct representing dataset
% This will probably do more later.
assert(ismatrix(pairs));
assert(size(pairs, 2) == 2);
unified.data = frame_data;
fst_cell = num2cell(int32(pairs(:, 1)))';
snd_cell = num2cell(int32(pairs(:, 2)))';
unified.pairs = struct('fst', fst_cell, 'snd', snd_cell);
unified.num_pairs = length(pairs);
unified.name = ds_name;
end

