import os
import scipy.io

def get_filenames( which_set):
    """Get file names for this set."""
    # if self._filenames is None:
    filenames = []
    split = scipy.io.loadmat(os.path.join('./', 'split.mat'))
    split = split[which_set + "Ids"]

    # To remove (Files with different size in img and mask)
    to_remove = [15188, ] + range(20803, 20835) + range(20858, 20861)

    for id in split:
        if id not in to_remove:
            filenames.append(str(id[0]).zfill(5) + '.png')
    # self._filenames = filenames
    print('GTA5: ' + which_set + ' ' + str(len(filenames)) + ' files')
    return filenames


if __name__ == '__main__':

    # hmm = scipy.io.loadmat('./mapping.mat')

    print("testing splitting mat")

    # g = GTA5()
    train=get_filenames('train')
    val=get_filenames('val')
    test=get_filenames('test')

    print("done")