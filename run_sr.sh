FALSE=0
TRUE=1

# assign global devices
export CUDA_VISIBLE_DEVICES='1'

#WORKROOT='workdir'
#NET_DATASET='EDSR'
#LOG_DIR=${WORKROOT}/${NET_DATASET}/log
#mkdir -p ${FULL_DIR} ${LOG_DIR}
#
#TIME_TAG=`date +"%Y%m%d_%H%M"`
#LOG_FILE=${LOG_DIR}/${TIME_TAG}.txt
#python -u main_sr.py 2>&1 | tee ${LOG_FILE}


# single image super resolution, SISR dataset
DATASET='DIV2K'
DATA_PATH='/home/zhaoyu/Data/DIV2K/'

# network model type and index
NET='EDSR'

# training parameters
NUM_EPOCH=120
BATCH_SIZE=16
BATCH_SIZE_TEST=1
STD_BATCH_SIZE=16
STD_INIT_LR=2e-4

BASIC_ARGUMENTS="--dataset ${DATASET}
                 --data_path ${DATA_PATH}
                 --net ${NET}
                 --num_epoch $((NUM_EPOCH))
                 --batch_size ${BATCH_SIZE}
                 --batch_size_test ${BATCH_SIZE_TEST}
                 --std_batch_size ${STD_BATCH_SIZE}
                 --std_init_lr ${STD_INIT_LR}"

NET_DATASET=${NET}${NET_INDEX}_${DATASET}
WORKROOT='workdir'

# append working directories arguments
FULL_DIR=${WORKROOT}/${NET_DATASET}/full
LOG_DIR=${WORKROOT}/${NET_DATASET}/log
mkdir -p ${FULL_DIR} ${LOG_DIR}
DIR_ARGUMENTS=" --full_dir ${FULL_DIR} --log_dir ${LOG_DIR} "
BASIC_ARGUMENTS+=${DIR_ARGUMENTS}

# prune switch
PRUNE_FLAG=${TRUE}
PRUNE_ARGUMENTS=" --prune_flag ${PRUNE_FLAG} "
if [ ${PRUNE_FLAG} == ${TRUE} ]; then
	SLIM_DIR=${WORKROOT}/${NET_DATASET}/slim
	WARMUP_DIR=${WORKROOT}/${NET_DATASET}/warmup
	SEARCH_DIR=${WORKROOT}/${NET_DATASET}/search
	mkdir -p "${SLIM_DIR}" "${WARMUP_DIR}" "${SEARCH_DIR}"
	WEIGHT_FLOPS=0.01
	NUM_EPOCH_WARMUP=20
	NUM_EPOCH_SEARCH=20
	PRUNE_ARGUMENTS+="--weight_flops ${WEIGHT_FLOPS}
	                  --num_epoch_warmup ${NUM_EPOCH_WARMUP}
	                  --num_epoch_search ${NUM_EPOCH_SEARCH}
	                  --warmup_dir ${WARMUP_DIR}
		                --search_dir ${SEARCH_DIR}
		                --slim_dir ${SLIM_DIR}"
fi

BASIC_ARGUMENTS+=${PRUNE_ARGUMENTS}
echo python -u main.py ${BASIC_ARGUMENTS}
TIME_TAG=$(date +"%Y%m%d_%H%M")
LOG_FILE=${LOG_DIR}/${TIME_TAG}.txt
CMD="python -u main.py ${BASIC_ARGUMENTS}"
echo ${CMD} > ${LOG_FILE}
python -u main_sr.py ${BASIC_ARGUMENTS} 2>&1 | tee ${LOG_FILE}
