FALSE=0
TRUE=1

# assign global devices
N_GPU=1
export CUDA_VISIBLE_DEVICES='0'

# select from: ['cifar10', 'cifar100']
DATASET='cifar100'
DATA_PATH='/home/zhaoyu/Data/cifar100/'

# network model type and index
NET='resnet'
NET_INDEX=20

# training parameters
NUM_EPOCH=600
BATCH_SIZE=256
STD_BATCH_SIZE=256
STD_INIT_LR=1e-1
MOMENTUM=0.9
WEIGHT_DECAY=5e-4

BASIC_ARGUMENTS="--nproc ${N_GPU}
                 --dataset ${DATASET}
                 --data_path ${DATA_PATH}
                 --net ${NET}
                 --net_index $((NET_INDEX))
                 --num_epoch $((NUM_EPOCH))
                 --batch_size ${BATCH_SIZE}
                 --std_batch_size ${STD_BATCH_SIZE}
                 --std_init_lr ${STD_INIT_LR}
                 --momentum ${MOMENTUM}
                 --weight_decay ${WEIGHT_DECAY}"

NET_DATASET=${NET}${NET_INDEX}_${DATASET}
WORKROOT='workdir'

# append working directories arguments
FULL_DIR=${WORKROOT}/${NET_DATASET}/full
LOG_DIR=${WORKROOT}/${NET_DATASET}/log
mkdir -p ${FULL_DIR} ${LOG_DIR}
DIR_ARGUMENTS=" --full_dir ${FULL_DIR} --log_dir ${LOG_DIR} "
BASIC_ARGUMENTS+=${DIR_ARGUMENTS}

# distillation switch
DST_FLAG=${FALSE}
DST_ARGUMENTS=" --dst_flag ${DST_FLAG} "
if [ ${DST_FLAG} == ${TRUE} ]; then
	TEACHER_NET='resnet'
	TEACHER_NET_INDEX=$((NET_INDEX))
	DST_TEMPERATURE=2
	DST_LOSS_WEIGHT=4
	TEACHER_DIR=${WORKROOT}/${NET_DATASET}/teacher
	mkdir -p ${TEACHER_DIR}
	if [ "`ls -A ${TEACHER_DIR}`" = "" ]; then
		echo "${TEACHER_DIR} is empty, load model from ${FULL_DIR}"
		cp ${FULL_DIR}/* ${TEACHER_DIR}
	fi
	DST_ARGUMENTS+="--teacher_net ${TEACHER_NET}
                  --teacher_net_index ${TEACHER_NET_INDEX}
                  --dst_temperature ${DST_TEMPERATURE}
                  --dst_loss_weight ${DST_LOSS_WEIGHT}
                  --teacher_dir ${TEACHER_DIR}"
fi
BASIC_ARGUMENTS+=${DST_ARGUMENTS}

# prune switch
PRUNE_FLAG=${FALSE}
PRUNE_ARGUMENTS=" --prune_flag ${PRUNE_FLAG} "
if [ ${PRUNE_FLAG} == ${TRUE} ]; then
	SLIM_DIR=${WORKROOT}/${NET_DATASET}/slim
	WARMUP_DIR=${WORKROOT}/${NET_DATASET}/warmup
	SEARCH_DIR=${WORKROOT}/${NET_DATASET}/search
	mkdir -p ${SLIM_DIR} ${WARMUP_DIR} ${SEARCH_DIR}
	WEIGHT_FLOPS=1
	NUM_EPOCH_WARMUP=150
	NUM_EPOCH_SEARCH=150
	PRUNE_ARGUMENTS+="--weight_flops ${WEIGHT_FLOPS}
	                  --num_epoch_warmup ${NUM_EPOCH_WARMUP}
	                  --num_epoch_search ${NUM_EPOCH_SEARCH}
	                  --warmup_dir ${WARMUP_DIR}
		                --search_dir ${SEARCH_DIR}
		                --slim_dir ${SLIM_DIR}"
fi
BASIC_ARGUMENTS+=${PRUNE_ARGUMENTS}


TIME_TAG=`date +"%Y%m%d_%H%M"`
LOG_FILE=${LOG_DIR}/${TIME_TAG}.txt
#echo python -u main.py ${BASIC_ARGUMENTS}

if (( ${N_GPU} > 1 )); then
  CMD="python -u -m torch.distributed.launch --nproc_per_node ${N_GPU} main.py ${BASIC_ARGUMENTS}"
  echo ${CMD}
  echo ${CMD} > ${LOG_FILE}
  python -u -m torch.distributed.launch --nproc_per_node ${N_GPU} main.py ${BASIC_ARGUMENTS} 2>&1 | tee -a ${LOG_FILE}
else
  CMD="python -u main.py ${BASIC_ARGUMENTS}"
  echo ${CMD}
  echo ${CMD} > ${LOG_FILE}
  python -u main.py ${BASIC_ARGUMENTS} 2>&1 | tee -a ${LOG_FILE}
fi
