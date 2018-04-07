import os


# ======= data path

def data_path():
    """
    1) tries to get data path from os environments $DATASETS
    2) metacentrum
    3) pchradis

    :param dataset: subfolder in data path (e.g 'gta' or 'CamVid')
    :rtype str:
    """
    try:
        data_path = os.environ['DATASETS']
    except:
        home_user = os.environ['USER']
        if home_user == 'mlyko':
            try:
                # metacentrum try
                if os.environ['WHICH_SERVER'] == 'metacentrum':
                    data_path = '/storage/brno7-cerit/home/mlyko/data/'
            except:
                data_path = '/home/mlyko/data/'
        elif home_user == 'mlykotom':
            data_path = '/Volumes/mlydrive/data/'
        elif home_user == 'xmlyna06':
            data_path = '/home/xmlyna06/data/'
        else:
            raise Exception("Unknown data path!")

    return data_path


def target_size():
    # return 360, 648
    # return 384, 640
    # return 288, 480
    # return (1052, 1914) # original
    return 256, 512


def weights_path():
    return '/home/mlyko/weights/'
