function det = BUFFY_transback(boxes)

% -------------------
% Generate candidate keypoint locations
% Our model produce 18 keypoint locations including joints and their middle points
% But for BUFFY evaluation, we will only use the original 10 joints
I = [1 2 3 4 5  6  7 8  9  10];
J = [1 2 3 5 11 13 7 15 10 18];
A = [1 1 1 1 1  1  1 1  1  1];
Transback = full(sparse(I,J,A,10,18));

det = struct('point', cell(1, numel(boxes)), 'score', cell(1, numel(boxes)));
for n = 1:length(boxes)
  if isempty(boxes{n}), continue, end;
  box = boxes{n};
  b = box(:, 1:floor(size(box, 2)/4)*4);
  b = reshape(b, size(b,1), 4, size(b,2)/4);
  b = permute(b,[1 3 2]);
  bx = .5*b(:,:,1) + .5*b(:,:,3);
  by = .5*b(:,:,2) + .5*b(:,:,4);
  for i = 1:size(b,1)
    det(n).point(:,:,i) = Transback * [bx(i,:)' by(i,:)'];
    det(n).score(i) = box(i, end);
  end
end