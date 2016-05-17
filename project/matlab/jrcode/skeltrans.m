function rv = skeltrans(orig_joints, trans_spec)
%SKELTRANS Use a translation spec to interpolate new skeleton
% Orig_joints will just be j*2 matrix, trans_spec is j'-element struct
% array with .indices and .weights attributes. trans_spec(i).indices gives
% the indices of source joints used to calculate the i-th dest joint,
% trans_spec(i).weights contains a weight for each of those indices (should
% sum to 1!).

% TODO: Try to translate a whole dataset of joints at once (although
% probably still iterating over trans_spec one at a time). I think that
% could well work and might be a lot faster.
assert(ismatrix(orig_joints) && size(orig_joints, 2) == 2);
assert(isstruct(trans_spec) && hasfield(trans_spec, 'indices') ...
    && hasfield(trans_spec, 'weights'));
rv = zeros(length(trans_spec), 2);
for jprime=1:length(trans_spec)
    jpw = trans_spec(jprime).weights;
    jpi = trans_spec(jprime).indices;
    assert(isvector(jpw) && length(jpw) == length(jpi));
    jpw = jpw(:); % Force into column vector
    assert(abs(sum(jpw) - 1) < 1e-5 && all(0 <= jpw & jpw <= 1), ...
        'Weights need to be around 1');
    
    sources = orig_joints(jpi, :);
    scaled = bsxfun(@times, sources, jpw);
    rv(jprime, :) = sum(scaled, 1);
end
end
