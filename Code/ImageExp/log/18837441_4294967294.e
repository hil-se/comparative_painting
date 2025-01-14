[1;31m==>[0m Error: py-keras matches multiple packages.
  Matching packages:
    [0;90mjsj35ex[0m py-keras[0;36m@3.2.1[0m[0;32m%gcc[0m[0;32m@12.3.1[0m[0;35m arch=linux-rhel9-skylake_avx512[0m
    [0;90mukcde6j[0m py-keras[0;36m@3.2.1[0m[0;32m%gcc[0m[0;32m@12.3.1[0m[0;35m arch=linux-rhel9-skylake_avx512[0m
    [0;90m7b6blx5[0m py-keras[0;36m@3.2.1[0m[0;32m%gcc[0m[0;32m@12.3.1[0m[0;35m arch=linux-rhel9-skylake_avx512[0m
  Use a more specific spec (e.g., prepend '/' to the hash).
2024-07-15 15:21:21.991492: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38367 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:3b:00.0, compute capability: 8.0
2024-07-15 15:21:29.588933: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel/dropout/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-15 15:21:30.149377: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:432] Loaded cuDNN version 8700
2024-07-15 15:21:32.937888: I tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:606] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2024-07-15 15:21:34.068519: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7fa9f7723ae0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2024-07-15 15:21:34.068557: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB, Compute Capability 8.0
2024-07-15 15:21:34.147141: I ./tensorflow/compiler/jit/device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
2024-07-15 16:07:19.353837: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape indual_encoder_all/model_1/dropout_2/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-15 17:13:46.696922: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel_2/dropout_4/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-15 17:58:14.108666: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape indual_encoder_all_1/model_3/dropout_6/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-15 19:25:34.050939: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel_4/dropout_8/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
/var/spool/slurmd/job18837441/slurm_script: line 48: 880015 Killed                  python3 Experiments.py
slurmstepd: error: Detected 1 oom_kill event in StepId=18837441.batch. Some of the step tasks have been OOM Killed.
