[1;31m==>[0m Error: py-keras matches multiple packages.
  Matching packages:
    [0;90mjsj35ex[0m py-keras[0;36m@3.2.1[0m[0;32m%gcc[0m[0;32m@12.3.1[0m[0;35m arch=linux-rhel9-skylake_avx512[0m
    [0;90mukcde6j[0m py-keras[0;36m@3.2.1[0m[0;32m%gcc[0m[0;32m@12.3.1[0m[0;35m arch=linux-rhel9-skylake_avx512[0m
    [0;90m7b6blx5[0m py-keras[0;36m@3.2.1[0m[0;32m%gcc[0m[0;32m@12.3.1[0m[0;35m arch=linux-rhel9-skylake_avx512[0m
  Use a more specific spec (e.g., prepend '/' to the hash).
2024-07-16 00:55:37.832992: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38367 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:3b:00.0, compute capability: 8.0
2024-07-16 00:55:45.104608: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel/dropout/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-16 00:55:45.521189: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:432] Loaded cuDNN version 8700
2024-07-16 00:55:48.144251: I tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:606] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2024-07-16 00:55:48.425761: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f635f8f5be0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2024-07-16 00:55:48.425802: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB, Compute Capability 8.0
2024-07-16 00:55:48.504194: I ./tensorflow/compiler/jit/device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
2024-07-16 01:22:16.009331: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape indual_encoder_all/model_1/dropout_2/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-16 02:51:42.499450: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel_2/dropout_4/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-16 03:14:39.648225: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape indual_encoder_all_1/model_3/dropout_6/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-16 04:25:50.810516: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel_4/dropout_8/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-16 04:51:36.499972: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape indual_encoder_all_2/model_5/dropout_10/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
