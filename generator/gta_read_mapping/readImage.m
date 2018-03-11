frameId  = 17086;
% frameId  = 16116;

% make filenames from frame id
imageFilename = sprintf('/storage/brno7-cerit/home/mlyko/data/gta/images/%05d.png', frameId);
labelFilename = sprintf('/storage/brno7-cerit/home/mlyko/data/gta/labels/%05d.png', frameId);

img = imread(imageFilename);

% the labels are stored as class ids with a color mapping.
% to get the current color mapping from a file use
[labels, currentMapping] = imread(labelFilename);

% load mapping for CamVid and CityScapes
mapping = load('mapping.mat', 'cityscapesMap', 'camvidMap', 'classes');

figure; 
imshow(img);
title(sprintf('Image %d', frameId));

figure; 
imshow(labels, mapping.cityscapesMap);
title('Labels (CamVid colors)');

figure; 
imshow(labels, mapping.camvidMap);
title('Labels (CamVid colors)');
