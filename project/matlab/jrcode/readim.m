function im = readim(datum)
%READIM Read the image associated with a datum.
% Handles H3.6M-style videos as well as images (so it doesn't matter which
% dataset the datum is from).
if hasfield(datum, 'image_path')
    % Just read from image path
    im = imread(datum.image_path);
elseif hasfield(datum, 'frame_time')
    % XXX: This is hacky and terrible. It will break as soon as I move
    % h3.6m
    vid_path = sprintf('datasets/h3.6m/S%i/Videos/%s.%i.mp4', ...
        datum.subject, datum.action, datum.camera);
    im = mp4_imread(vid_path, datum.frame_time);
else
    throw(MException('JointRegressor:readim:noFrame', 'Don''t know how to read this datum'));
end
