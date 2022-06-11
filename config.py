# dataset info
DATASETS = ['eth_hotel', 'eth_univ', 'ucy_univ', 'ucy_zara01', 'ucy_zara02', 'aldi', 'edeka', 'globus', 'rewe',
            'area_aldi_checkout', 'area_aldi_hall', 'area_globus_checkout', 'area_globus_hall',
            'area_aldi_wm', 'area_aldi_sa', 'area_globus_wm', 'area_globus_sa']
DATASETS_RETAIL = ['aldi', 'edeka', 'globus', 'rewe',
					'area_aldi_checkout', 'area_aldi_hall', 'area_globus_checkout', 'area_globus_hall',
					'area_aldi_wm', 'area_aldi_sa', 'area_globus_wm', 'area_globus_sa']

DATASETS_FIRST_LOAD = []
DATASET_DELIMITER = ';'

# dataset/model loading
#DATASETS_TRAIN = ['eth_hotel', 'eth_univ', 'ucy_univ', 'ucy_zara01']#['eth_hotel', 'eth_univ', 'ucy_univ', 'ucy_zara01', 'ucy_zara02']
#DATASET_TEST = 'ucy_zara02'
DATASETS_TRAIN = ['globus']
DATASET_TEST = 'globus'
MODEL_PATH = 'models/model_edeka_lstm_leave.h5'
LOAD_MODEL = True
MODEL_TYPE = 'lstm'         #'dense', 'conv', 'lstm', 'gan'

# dataset filtering
X_MIN = -100
X_MAX = 100     #44.56
Y_MIN = -100
Y_MAX = 100     #26.40
#START_DATE = '2019-08-28 01:00:00'      # wed mor
#END_DATE = '2019-08-28 13:00:00'
START_DATE = '2019-08-03 13:00:00'     # sat aft
END_DATE = '2019-08-03 23:00:00'
#START_DATE = '2019-07-10 00:00:00'
#END_DATE = '2019-09-11 00:00:00'
MIN_SEQ_STD = 0.1
MIN_SEQ_LEN = 20

# dataset preparation and training
VARS = ['x', 'y']
SEQ_LEN = 20          # <= MIN_SEQ_LEN
N_FEATURES = 8        # < SEQ_LEN
AUGMENT = False
SHUFFLE = True
REAPPLY_FILTER = True                  #TODO True
TRAIN_TEST_RATIO = 0.8
VAL_TRAIN_RATIO = 0.2
N_EPOCHS = 10000
PATIENCE = 200
BATCH_SIZE = 64                          #TODO 64
DROPOUT = [0, 0]
#N_NEURONS = [512, 256, 128, 64, 32]        #dense
#N_NEURONS = [64, 16, 32, 64]               #conv
N_NEURONS = [64, 64, 32]                    #lstm
