2024-07-12 03:50:30.349444: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38367 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:3b:00.0, compute capability: 8.0
2024-07-12 03:50:34.404140: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel/dropout/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-12 03:50:34.804606: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:432] Loaded cuDNN version 8700
2024-07-12 03:50:35.696368: I tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:606] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2024-07-12 03:50:35.995944: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f42a4002950 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2024-07-12 03:50:35.995985: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB, Compute Capability 8.0
2024-07-12 03:50:36.095853: I ./tensorflow/compiler/jit/device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
Traceback (most recent call last):
  File "/home/xx4455/Comparable/Code/ImageExp/Experiments.py", line 104, in <module>
    experiment(dataName="FaceImage", col='Average', height=250, width=250)
  File "/home/xx4455/Comparable/Code/ImageExp/Experiments.py", line 27, in experiment
    precision_r, recall_r) = cl.regressionExperiment(
  File "/home/xx4455/Comparable/Code/ImageExp/Classification.py", line 310, in regressionExperiment
    return m.mse(), m.r2(), m.pearsonr_coefficient(), m.pearsonr_value(), m.spearmanr_coefficient(), m.spearmanr_value(), m.MI_con_info(
  File "/home/xx4455/Comparable/Code/ImageExp/metrics.py", line 39, in pearsonr_coefficient
    return pearsonr(self.y, self.y_pred).r
AttributeError: 'tuple' object has no attribute 'r'
