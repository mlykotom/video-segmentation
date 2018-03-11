# coding=utf-8
# !/usr/bin/python3
import errno
import os
import re
import shutil
import sys
from random import shuffle

HOW_MANY_PREV = 3
DATASET_LENGTH = 5000
SHOULD_COPY = HOW_MANY_PREV * DATASET_LENGTH


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def get_all_files(dataset_files_exported='./trainvaltest.txt'):
    file_pattern = re.compile("\./(?:train|val|test)/(?:[^/]*)/(?:[^/]*)_(?:[^_]+)_(?P<frame>[^_]+)_leftImg8bit\.png")

    print_progress(0, SHOULD_COPY, '1) getting files')

    found = 0
    file_paths = []
    with open(dataset_files_exported, 'r') as fp:
        # files_num = 0
        for line in fp:
            path = line.rstrip('\n')
            # /test/munich/munich_000288_000019_leftImg8bit.png

            match = file_pattern.match(path)
            if match is None:
                print("skipping path %s" % path)
                continue

            match_dict = match.groupdict()
            frame_i = int(match_dict['frame'])

            for i in range(frame_i - HOW_MANY_PREV, frame_i):
                frame_str = str(i).zfill(6)

                file_to_copy = path.replace(match_dict['frame'], frame_str)
                file_paths.append(file_to_copy)
                found += 1
                print_progress(found, SHOULD_COPY, '1) getting files')

        print("files pattern found %d" % found)
    return file_paths


def copy_files(files_to_copy, target='../city_sel_3'):
    print_progress(0, SHOULD_COPY, '2) copying files')
    shuffle(files_to_copy)

    for i, f_from in enumerate(files_to_copy):
        print_progress(i, SHOULD_COPY, '2) copying files')
        f_to = target + f_from.lstrip('.')
        if not os.path.isfile(f_to):
            # print("copying %s" % f_to)
            try:
                shutil.copy(f_from, f_to)
            except IOError as e:
                if e.errno != errno.ENOENT:
                    raise
                os.makedirs(os.path.dirname(f_to), exist_ok=True)
                shutil.copy(f_from, f_to)


if __name__ == '__main__':
    files = get_all_files()
    copy_files(files)
