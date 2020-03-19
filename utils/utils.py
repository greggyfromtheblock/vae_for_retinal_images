import logging
import argparse
import json
import os
import sys
import torch


def set_up_logger(out_directory, name, to_console=True, console_level='DEBUG'):
    """
    Set up the loggers for the module. Also set up filehandlers and streamhandlers (to console).
    """
    # get a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    logfile = os.path.join(out_directory, '{}.log'.format(name))
    if os.path.isfile(logfile):
        with open(logfile, 'a') as ofile:
            ofile.write('\n' + '=' * 50 + '\n\n\n\n\n')

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('{asctime:.16}-{name}-{levelname:.1}: {message}', style='{')  # ({process})

    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)

    if to_console:
        # create console handler with a higher log level
        levels = {'DEBUG': logging.DEBUG,
                  'INFO': logging.INFO,
                  'WARNING': logging.WARNING}

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(levels[console_level])
        logger.addHandler(ch)
        ch.setFormatter(formatter)

    return logger


def set_default_options(config_file, parser):
    """
    Read the config.JSON and set the default options accordingly.
    :return:
    """

    with open(config_file, 'r') as open_config_file:
        config_dict = json.load(open_config_file)

    def str2bool(x):
        if x.lower() in ['true', 'yes', 't', 'y']:
            return True
        elif x.lower() in ['false', 'no', 'f', 'n']:
            return False
        else:
            raise argparse.ArgumentError('{} is not a valid bool.'.format(x))

    # define a mapping from string to callable type:
    str2type = {'str': str,
                'int': int,
                'float': float,
                'bool': str2bool}

    for arg in config_dict.keys():
        default = config_dict[arg]['value']
        if torch:
            if torch.cuda.is_available() and 'value_gpu' in config_dict[arg]:
                default = config_dict[arg]['value_gpu']

        if config_dict[arg]['type'] == 'bool':
            if default.strip() == 'True':
                default = True
            elif default.strip() == 'False':
                default = False
            else:
                raise Exception("Bool Argument must be 'True' or 'False'.")

        if config_dict[arg]['short']:
            parser.add_argument(arg, config_dict[arg]['short'],
                                default=default,
                                help=config_dict[arg]['help'],
                                type=str2type[config_dict[arg]['type']],
                                )
        else:
            parser.add_argument(arg,
                                default=default,
                                help=config_dict[arg]['help'],
                                type=str2type[config_dict[arg]['type']],
                                )


def setup(running_script, config, args=None, logging_name=None):
    """
    For all command-line exposed scripts. Prepares logger and arguments parser
    :return:
    """
    # Get parser
    parser = argparse.ArgumentParser(description='Options to run {}'.format(running_script))

    # set config
    set_default_options(config_file=config,
                        parser=parser)
    FLAGS = parser.parse_args(args)

    # save command line arguments
    path = os.path.join(FLAGS.out_directory, 'FLAGS.json')
    # don't overwrite old ones!
    if False:
        if os.path.exists(path):
            count = 1
            while os.path.exists(os.path.join(FLAGS.out_directory, 'FLAGS_{}.json'.format(count))):
                count += 1
            path = os.path.join(FLAGS.out_directory, 'FLAGS_{}.json'.format(count))
        with open(path, 'w') as ofile:
            json.dump(vars(FLAGS), ofile)

    # get logger
    if not logging_name:
        logging_name = FLAGS.logging_name
    logger = set_up_logger(FLAGS.out_directory, logging_name)

    logger.info('Running {} in process {}.'.format(running_script, os.getpid()))
    logger.info('Writing to {}'.format(FLAGS.out_directory))
    logger.info('Called with the following arguments: \n{}\nfrom\n{}\n'.format(', '.join(sys.argv), os.getcwd()))

    # log the FLAGS actually used for ths run
    out = 'Initialized with FLAGS:\n'
    tmp = FLAGS.__dict__
    params = list(tmp.keys())
    m_len = max([len(p) for p in params])
    for p in sorted(tmp):
        out += '{:<{prec}} = {},\n'.format(p, tmp[p], prec=m_len)
    logger.info(out.replace("'", "").replace('[', '').replace(']', ''))

    return FLAGS, logger