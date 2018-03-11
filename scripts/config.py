import argparse
import os
from collections import OrderedDict


def str2bool(string):
    return string.lower() in ['yes', 'true', 't', 1]


def Config():
    parser = argparse.ArgumentParser(description="Collecting Data")
    parser.register('type', 'bool', str2bool)
    # input parsing parameters
    parser.add_argument('-T', '--TRAIN_FILE_PATH', type=str, default='', help='train file path')
    parser.add_argument('-V', '--VAL_FILE_PATH', type=str, default='', help='val file path')
    parser.add_argument('-E', '--TEST_FILE_PATH', type=str, default='', help='test file path')
    parser.add_argument('-P', '--PRE_TRAINED_PATH', type=str, default='', help='word2vec file path')
    parser.add_argument('-t', '--TRAIN_SIZE', type=int, default=0, help='train size')
    parser.add_argument('-v', '--VAL_SIZE', type=int, default=0, help='validation size')
    parser.add_argument('-e', '--TEST_SIZE', type=int, default=0, help='test size')
    parser.add_argument('--TOTAL_SET_PATH', type=str, default='', help='total set path')
    parser.add_argument('--MY_TEST_FILE_PATH', type=str, default='', help='new test file path')
    parser.add_argument('--TAG_SCHEME', type=str, default='iob', help='iob or iobes')
    # model parsing parameters
    # embedding parameters
    parser.add_argument('-w', '--WORD_INPUT_DIM', type=int, default=300, help='word embedding dimension')
    parser.add_argument('-W', '--WORD_HIDDEN_DIM', type=int, default=512, help='word hidden dimension')
    parser.add_argument('-b', '--WORD_BIDIRECT', type='bool', default=True, help='whether to use word bidirect lstm')
    parser.add_argument('-f', '--FINE_TUNE', type='bool', default=True, help='whether to use fine tuning for w2v')
    # LSTM parameters
    parser.add_argument('-c', '--CHAR_INPUT_DIM', type=int, default=0, help='char embedding dimension')
    parser.add_argument('-C', '--CHAR_HIDDEN_DIM', type=int, default=0, help='char hidden dimension')
    parser.add_argument('-B', '--CHAR_BIDIRECT', type='bool', default=True, help='whether to use char bidirect lstm')
    parser.add_argument('--NUM_RNN_LAYERS', type=int, default=1, help='number of layers of rnn')
    # CNN parameters
    parser.add_argument('-i', '--FILTER_SIZE', type=int, default=3, help='filter for cov1d')
    parser.add_argument('-I', '--NUM_FILTERS', type=int, default=20, help='num of filters')
    #optimizer parameters
    parser.add_argument('-l', '--LEARNING_RATE', type=float, default=0.001, help='learning rate')
    parser.add_argument('-L', '--LEARNING_METHOD', type=str, default='sgd', help='learning method')
    parser.add_argument('-d', '--WEIGHT_DECAY', type=float, default=0, help='Weight decay')
    parser.add_argument('-n', '--CLIP_NORM', type=float, default=0, help='clip norm')
    parser.add_argument('-r', '--DROPPUT_OUT', type=float, default=0, help='droup out')
    parser.add_argument('--MOMENTUM', type=float, default=0.9)
    # training parameters
    parser.add_argument('-o', '--TRAIN_EPOCHS', type=int, default=1000, help='training epoch')
    parser.add_argument('-g', '--BATCH_SIZE', type=int, default=2, help='batch size')
    parser.add_argument('-q', '--FREQ_EVAL', type=int, default=10, help='frequency of evaluating on val+test set')
    parser.add_argument('-S', '--SAVE_MODEL_PATH', type=str, default='', help='the path to store best model')
    parser.add_argument('-R', '--RESTORE_MODEL_PATH', type=str, default='', help='the path to restore existing model')
    parser.add_argument('-G', '--TWO_LEVEL_CHAR_TAG', type='bool', default=False, help='whether to use two level character tagging')
    parser.add_argument('-m', '--RANDOM_SEED', type=int, default=42, help='random seed for shuffling data')
    parser.add_argument('--SAVE_PREDICT_PATH', type=str, default='')
    parser.add_argument('--SAVED_EPOCH', type=str, default='0', help='saved epoch')
    parser.add_argument('--WORD_LOWER', type='bool', default=True)
    parser.add_argument('--INSERT_SINGLETONS', type='bool', default=False)

    # model selection parameters
    parser.add_argument('-s', '--CHAR_SUPER', type='bool', default=False, help='whether to use char supervision or not')
    parser.add_argument('-H', '--HIERAR_BILSTM', type='bool', default=False, help='whether to use hierarchy bilstm or not')
    parser.add_argument('-N', '--CONNECTIONS', type=str, default='1', help='methods to connect char level and word level: 0 - no connection; 1 - connect char hidden to word lstm; 2 - connect char labels to word lstm; 3 - connect char hidden to word softmax; 4 - connect char labels to word softmax')
    parser.add_argument('-O', '--TRAIN_ORDER', type=str, default='0', help='training order: 0 - joint; 1 - in order; 2 - inverse order; 3 - alternative; 4 - onely char')
    parser.add_argument('-a', '--CHAR_ALONE_MODEL', type='bool', default=False, help='whether to use char alone model')
    parser.add_argument('-u', '--USE_WORD_INPUTS', type='bool', default=True, help='whether to use word inputs')
    parser.add_argument('-F', '--FLAG_SAVING', type=str, default='', help='flag for saving')
    parser.add_argument('-Y', '--USE_CRF_LOSS', type='bool', default=True, help='whether to use crf loss or softmax loss')
    parser.add_argument('-z', '--FROZEN', type='bool', default=False, help='whether to freeze char part when training word part')
    parser.add_argument('-x', '--TEST_ONLY', type='bool', default=False, help='test new data on saved models')
    parser.add_argument('-M', '--LM_VOCAB_PATH', type=str, default='', help='the path of lm vocabulary')
    parser.add_argument('-A', '--GAMMA', type=float, default=0.1, help='the variable to control the importatnce of LM')
    parser.add_argument('-U', '--OUTPUT_DIR', type=str, default='', help='name of output files for preprocessing of nuc')
    parser.add_argument('--LM_MODE', type=str, default='', help='nl, pl, both')
    parser.add_argument('--NL_FREQ', type=int, default=10000, help='extract NL vocab based on their frequency')
    parser.add_argument('--PL_FREQ', type=int, default=10000, help='extract PL vocab based on their frequency')
    parser.add_argument('--NL_COF', type=float, default=1.0, help='the coefficient for nl')
    parser.add_argument('--PL_COF', type=float, default=1.0, help='the coefficient for pl')
    parser.add_argument('--FIG_NAME', type=str, default='', help='the name of figure to store')
    parser.add_argument('--FIG_PATH', type=str, default='', help='the path to store the figure')
    parser.add_argument('--PATIENCES', type=int, default=3, help='patience for early stopping')
    parser.add_argument('--VERTICAL_CONNECT', type='bool', default=False, help='whether to connect hidden state of SL lstm to lm lstm')
    parser.add_argument('--PLACEHOLDER_TRAIN', type='bool', default=True, help='does the placeholder embedding trainable or not')
    parser.add_argument('--ADD_PL_PRIOR', type='bool', default=False, help='add pl prior auxiliary objective')
    parser.add_argument('--PRIOR_LAMBDA', type=float, default=1, help='factor for prior loss')
    parser.add_argument('--ADD_KEYWORDS', type='bool', default=False, help='add keywords loss')
    parser.add_argument('--KEYWORD_LAMBDA', type=float, default=1, help='keyword lambda')
    parser.add_argument('--USE_HIER_CHAR_LM', type='bool', default=False, help='use hierarchy character lm model')
    # parser.add_argument('--LM_LAMBDA', type=float, default=1.0)
    parser.add_argument('--CHAR_ATTENTION', type='bool', default=False)
    parser.add_argument('--CHAR_ENCODE', type=str, default='lstm', help='encoding character: lstm or cnn or None')
    args = parser.parse_args()

    print 'args: ', args

    # store parameters
    Model_Parameters = OrderedDict()
    # Model_Parameters['lm_lambda'] = args.LM_LAMBDA
    Model_Parameters['train_file_path'] = args.TRAIN_FILE_PATH
    Model_Parameters['val_file_path'] = args.VAL_FILE_PATH
    Model_Parameters['test_file_path'] = args.TEST_FILE_PATH
    Model_Parameters['train_size'] = args.TRAIN_SIZE
    Model_Parameters['val_size'] = args.VAL_SIZE
    Model_Parameters['test_size'] = args.TEST_SIZE
    Model_Parameters['total_set_path'] = args.TOTAL_SET_PATH
    Model_Parameters['tag_scheme'] = args.TAG_SCHEME

    Model_Parameters['num_layers'] = args.NUM_RNN_LAYERS
    Model_Parameters['pre_trained_path'] = args.PRE_TRAINED_PATH
    Model_Parameters['word_input_dim'] = args.WORD_INPUT_DIM
    Model_Parameters['word_hidden_dim'] = args.WORD_HIDDEN_DIM
    Model_Parameters['fine_tune_w2v'] = args.FINE_TUNE
    Model_Parameters['lr_rate'] = args.LEARNING_RATE
    Model_Parameters['lr_method'] = args.LEARNING_METHOD
    Model_Parameters['clip_norm'] = args.CLIP_NORM
    Model_Parameters['weight_decay'] = args.WEIGHT_DECAY
    Model_Parameters['momentum'] = args.MOMENTUM
    Model_Parameters['char_input_dim'] = args.CHAR_INPUT_DIM
    Model_Parameters['char_hidden_dim'] = args.CHAR_HIDDEN_DIM
    Model_Parameters['batch_size'] = args.BATCH_SIZE
    Model_Parameters['vocab_size'] = 0
    Model_Parameters['tag_size'] = 0
    Model_Parameters['char_vocab_size'] = 0
    Model_Parameters['filter_size'] = args.FILTER_SIZE
    Model_Parameters['num_filters'] = args.NUM_FILTERS
    Model_Parameters['max_sent_len'] = 0
    Model_Parameters['max_char_len'] = 0
    Model_Parameters['dropout'] = args.DROPPUT_OUT
    Model_Parameters['train_epochs'] = args.TRAIN_EPOCHS
    Model_Parameters['freq_eval'] = args.FREQ_EVAL
    Model_Parameters['char_tag_size'] = 0
    Model_Parameters['save_model_path'] = args.SAVE_MODEL_PATH
    if Model_Parameters['save_model_path'] and not os.path.exists(Model_Parameters['save_model_path']):
        os.makedirs(Model_Parameters['save_model_path'])
    Model_Parameters['restore_mode_path'] = args.RESTORE_MODEL_PATH
    Model_Parameters['random_seed'] = args.RANDOM_SEED

    # boolean parameters
    Model_Parameters['word_bidirect'] = args.WORD_BIDIRECT
    Model_Parameters['char_bidirect'] = args.CHAR_BIDIRECT
    Model_Parameters['use_word2vec'] = (args.PRE_TRAINED_PATH != '')
    # Model_Parameters['use_char_lstm'] = (args.CHAR_HIDDEN_DIM != 0)
    # Model_Parameters['use_char_cnn'] = (args.FILTER_SIZE != 0)
    Model_Parameters['add_char_super'] = args.CHAR_SUPER
    Model_Parameters['use_hierarchy_lstm'] = args.HIERAR_BILSTM
    Model_Parameters['two_level_char_tag'] = args.TWO_LEVEL_CHAR_TAG
    Model_Parameters['connection_method'] = args.CONNECTIONS
    Model_Parameters['train_order'] = args.TRAIN_ORDER
    Model_Parameters['use_char_alone_model'] = args.CHAR_ALONE_MODEL
    Model_Parameters['use_word_inputs'] = args.USE_WORD_INPUTS
    Model_Parameters['flag_saving'] = args.FLAG_SAVING
    Model_Parameters['frozen'] = args.FROZEN
    Model_Parameters['test_only'] = args.TEST_ONLY
    Model_Parameters['id_to_word_tag'] = {}
    Model_Parameters['id_to_char_tag'] = {}
    Model_Parameters['use_crf_loss'] = args.USE_CRF_LOSS
    Model_Parameters['lm_vocab_path'] = args.LM_VOCAB_PATH
    Model_Parameters['gamma'] = args.GAMMA
    Model_Parameters['start'] = '#<s>#'
    Model_Parameters['end'] = '#<e>#'
    Model_Parameters['unk'] = '<UNK>'
    # Model_Parameters['nl_placeholder'] = '#<nl_placeholder>#'
    # Model_Parameters['pl_placeholder'] = '#<pl_placeholder>#'
    Model_Parameters['output_dir'] = args.OUTPUT_DIR
    # Model_Parameters['lm_mode'] = args.LM_MODE
    # Model_Parameters['nl_freq'] = args.NL_FREQ
    # Model_Parameters['pl_freq'] = args.PL_FREQ
    # Model_Parameters['nl_cof'] = args.NL_COF
    # Model_Parameters['pl_cof'] = args.PL_COF
    Model_Parameters['fig_name'] = args.FIG_NAME
    Model_Parameters['fig_path'] = args.FIG_PATH
    Model_Parameters['patiences'] = args.PATIENCES
    # Model_Parameters['vertical_connect'] = args.VERTICAL_CONNECT
    # Model_Parameters['placeholder_train'] = args.PLACEHOLDER_TRAIN
    # Model_Parameters['add_pl_prior'] = args.ADD_PL_PRIOR
    # Model_Parameters['prior_lambda'] = args.PRIOR_LAMBDA
    # Model_Parameters['add_keywords'] = args.ADD_KEYWORDS
    # Model_Parameters['keyword_lambda'] = args.KEYWORD_LAMBDA
    Model_Parameters['my_test_file_path'] = args.MY_TEST_FILE_PATH
    Model_Parameters['save_predict_path'] = args.SAVE_PREDICT_PATH
    if Model_Parameters['save_predict_path'] and not os.path.exists(Model_Parameters['save_predict_path']):
        print Model_Parameters['save_predict_path']
        os.makedirs(Model_Parameters['save_predict_path'])
    Model_Parameters['saved_epoch'] = args.SAVED_EPOCH
    Model_Parameters['use_hier_char_lm'] = args.USE_HIER_CHAR_LM
    Model_Parameters['char_attention'] = args.CHAR_ATTENTION
    Model_Parameters['word_lower'] = args.WORD_LOWER
    Model_Parameters['insert_singletons'] = args.INSERT_SINGLETONS
    Model_Parameters['char_encode'] = args.CHAR_ENCODE
    return Model_Parameters
