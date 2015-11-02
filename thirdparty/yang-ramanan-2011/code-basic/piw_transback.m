function det = piw_transback(boxes)

% -------------------
% Generate candidate keypoint locations
Transback=1;

det = struct('point', cell(1, numel(boxes)), 'score', cell(1, numel(boxes)));
for n = 1:length(boxes)
    if isempty(boxes{n}), continue, end;
    % pose = reshape(boxes{n}(1:end-2), 4, 18).';
    box = boxes{n};
    b = box(:, 1:floor(size(box, 2)/4)*4);
    b = reshape(b, size(b,1), 4, size(b,2)/4);
    b = permute(b,[1 3 2]);
    bx = .5*b(:,:,1) + .5*b(:,:,3);
    by = .5*b(:,:,2) + .5*b(:,:,4);
    for i = 1:size(b,1)
        points = Transback * [bx(i,:)' by(i,:)'];
        points = cat(1, points, [0 0]);
        % Here's what the Buffy-trained detector gives us (XXX: This could be wrong!):
        % [Head, Neck, L-{shoulder, upper arm, elbow, lower arm, wrist,
        % upper torso, mid torso, hip}, R-{shoulder, upper arm, elbow,
        % lower arm, wrist, upper torso, mid torso, hip}]
        % Here's what PIW contains (observer perspective):
        % [Head, R-{shoulder, upper arm, elbow, lower arm, wrist},
        % L-{shoulder, upper arm, elbow, lower arm, wrist}, {upper, lower}
        % torso midpoint]
        % In the below, I use 19 as a synonym for "missing"
        piw_map = [19, 11, 12, 13, 14, 15, 3, 4, 5, 6, 7, 19, 19];
        det(n).point(:,:,i) = points(piw_map, :);
        det(n).score(i) = box(i, end);
    end
end
end



