import json


# TODO prepare - on epoch end to save to this file, on start check this file? (some arg maybe)
def parse_checkpoint(model_name):
    checkpoint_filename = '%s_last_checkpoint.json' % model_name

    with open(checkpoint_filename) as f:
        data = json.load(f)

    print(data)


def last_checkpoint_saver(model_name):
    """
    TODO return callback for keras fit
    This will be called on every epoch end and saves actual
    file of weights with last epoch to json.
    When resuming, it will look into the json and start from previously ended epoch.
    :param model_name:
    :return:
    """
    pass


if __name__ == '__main__':
    # parse_checkpoint('SegNet')

    fp = open('MobileUNet.last_epoch.json', 'w')
    json.dump({"epoch": 0}, fp)
    fp.close()
