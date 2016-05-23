% some function to return a unique id for a frame. unique in the dataset.
function frameid = get_framenum(imname)
% this just extracts the number at the end of the filename (e.g.
% 'file-042-00013.png' --> 13)
frame_match = regexp(imname, '[^\d](?<frame_no>\d+)\.png$', 'names');
assert(numel(frame_match) == 1);
str = frame_match.frame_no;
%str = strtok(strrep(imname, '-', ''), '.');
%str = str(str<'A');
frameid = str2double(str);
end
