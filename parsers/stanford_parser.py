from __future__ import print_function
import subprocess
import os
import logging
import multiprocessing
from manage import app


__author__ = 'suraj'

__all__ = ['run_stanford_parser',
           'not_parsed_files', 'standford_parser']

logger = logging.getLogger("standord-parser")
logger.setLevel(logging.INFO)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)

STANFORD_LEXPARSER = app.config['STANFORD_PARSER']


def standford_parser(file_name, output_dir):
    dir_name, base_name = os.path.split(file_name)

    try:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        if os.path.exists(os.path.join(output_dir, os.path.splitext(base_name)[0] + "_st_parser.txt")):
            logger.info("Already parsed %s" % os.path.splitext(base_name)[0] + "_st_parser.txt")
            return os.path.splitext(base_name)[0] + "_st_parser.txt"
        else:
            env = dict(os.environ)
            env['LC_ALL'] = 'en_US.UTF-8'
            cmd_args = {'st_parser': STANFORD_LEXPARSER, 'input_file': '"' + file_name + '"',
                        'output_file': '"' + os.path.join(output_dir, os.path.splitext(base_name)[0] +
                                                          "_st_parser.txt") + '"'}
            cmd = "sh {st_parser}  {input_file} >  {output_file} ".format(**cmd_args)
            print(cmd)
            subprocess.check_call(cmd, shell=True, env=env)
            return os.path.splitext(base_name)[0] + "_st_parser.txt"
    except Exception as e:
        logger.fatal("Error for file --> %s ,%s" % (dir_name, base_name))
        return None


def produce_task(queue, dir_path, output_dir):
    def add_files(done_lst):
        for file in os.listdir(dir_path):
            if file.replace('.txt', '_st_parser.txt') in done_lst:
                continue
            queue.put(os.path.join(dir_path, file))
            logger.info("producer [%s] putting value [%s] in queue..." % (multiprocessing.current_process().name, file))

    files_done = os.listdir(output_dir)
    add_files(files_done)


def consumer_task(queue, func, output_dir):
    while not queue.empty():
        work_args = queue.get(True, 0.1)
        logger.info(
            "consumer [%s] getting value [%s] from queue..." % (multiprocessing.current_process().name, work_args))
        func(work_args, output_dir)


def not_parsed_files(input_dir, output_dir, suffix):
    files = os.listdir(input_dir)
    parsed_files = [file.replace(suffix, '.txt') for file in os.listdir(output_dir)]
    remaining_files = set(files).difference(set(parsed_files))
    print("Remaining files {}".format(len(remaining_files)))
    print(remaining_files)


def run_stanford_parser(data_dir):
    # data_dir = app.config['DATA_DIR']
    output_dir = app.config['STANFORD_PARSER_OUTPUT']
    manager = multiprocessing.Manager()
    data_queue = manager.Queue()

    # producer
    producer = multiprocessing.Process(target=produce_task, args=(data_queue, data_dir, output_dir))
    producer.start()
    producer.join()

    logger.info("Number of jobs %s", data_queue.qsize())

    # consumers
    consumer_list = []
    for i in range(multiprocessing.cpu_count()):
        consumer = multiprocessing.Process(target=consumer_task, args=(data_queue,
                                                                       standford_parser, output_dir))
        consumer_list.append(consumer)
        consumer.start()

    [consumer.join() for consumer in consumer_list]

    logger.info("Done ...")
