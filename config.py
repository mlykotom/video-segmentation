import os


# ======= data path

def data_path(dataset=None):
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
            data_path = '/storage/brno7-cerit/home/mlyko/data/'
        elif home_user == 'xmlyna06':
            data_path = '/home/xmlyna06/data/'
        else:
            raise Exception("Unknown data path!")

    if dataset is not None:
        data_path = os.path.join(data_path, dataset, '')

    return data_path
