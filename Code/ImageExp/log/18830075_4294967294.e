2024-07-12 03:55:32.660873: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38367 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:3b:00.0, compute capability: 8.0
2024-07-12 03:55:35.188166: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel/dropout/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-12 03:55:35.421777: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:432] Loaded cuDNN version 8700
2024-07-12 03:55:36.264144: I tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:606] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2024-07-12 03:55:36.299982: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f54fb8839d0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2024-07-12 03:55:36.300023: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB, Compute Capability 8.0
2024-07-12 03:55:36.403575: I ./tensorflow/compiler/jit/device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
2024-07-12 03:55:59.629165: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape indual_encoder_all/model_1/dropout_2/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
Traceback (most recent call last):
  File "/home/xx4455/Comparable/Code/ImageExp/Experiments.py", line 104, in <module>
    experiment(dataName="FaceImage", col='Average', height=250, width=250)
  File "/home/xx4455/Comparable/Code/ImageExp/Experiments.py", line 63, in experiment
    MI_encoder_sex_single) = cl.comparabilityExperiment(
  File "/home/xx4455/Comparable/Code/ImageExp/Classification.py", line 231, in comparabilityExperiment
    recall, precision, F1, accuracy, AOD_race = test_model(test, dual_encoder, protected_ts_AB_race)
  File "/home/xx4455/Comparable/Code/ImageExp/Classification.py", line 94, in test_model
    return evaluate(labels, predictions, protected)
  File "/home/xx4455/Comparable/Code/ImageExp/Classification.py", line 160, in evaluate
    FPR_i_j = FP_i_j / F_i_j
ZeroDivisionError: division by zero
