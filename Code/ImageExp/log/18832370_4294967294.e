[1;31m==>[0m Error: py-keras matches multiple packages.
  Matching packages:
    [0;90mjsj35ex[0m py-keras[0;36m@3.2.1[0m[0;32m%gcc[0m[0;32m@12.3.1[0m[0;35m arch=linux-rhel9-skylake_avx512[0m
    [0;90mukcde6j[0m py-keras[0;36m@3.2.1[0m[0;32m%gcc[0m[0;32m@12.3.1[0m[0;35m arch=linux-rhel9-skylake_avx512[0m
    [0;90m7b6blx5[0m py-keras[0;36m@3.2.1[0m[0;32m%gcc[0m[0;32m@12.3.1[0m[0;35m arch=linux-rhel9-skylake_avx512[0m
  Use a more specific spec (e.g., prepend '/' to the hash).
2024-07-12 13:39:32.798552: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38367 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:5e:00.0, compute capability: 8.0
2024-07-12 13:39:40.232262: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel/dropout/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-12 13:39:40.640419: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:432] Loaded cuDNN version 8700
2024-07-12 13:39:42.413161: I tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:606] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2024-07-12 13:39:42.751737: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f588914d610 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2024-07-12 13:39:42.751789: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB, Compute Capability 8.0
2024-07-12 13:39:42.831807: I ./tensorflow/compiler/jit/device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
2024-07-12 14:19:51.116151: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape indual_encoder_all/model_1/dropout_2/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-12 14:20:01.744246: W tensorflow/tsl/framework/bfc_allocator.cc:485] Allocator (GPU_0_bfc) ran out of memory trying to allocate 488.28MiB (rounded to 512000000)requested by op dual_encoder_all/model_1/conv2d_17/Relu_1
If the cause is memory fragmentation maybe the environment variable 'TF_GPU_ALLOCATOR=cuda_malloc_async' will improve the situation. 
Current allocation summary follows.
Current allocation summary follows.
2024-07-12 14:20:01.744313: I tensorflow/tsl/framework/bfc_allocator.cc:1039] BFCAllocator dump for GPU_0_bfc
2024-07-12 14:20:01.744328: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (256): 	Total Chunks: 85, Chunks in use: 85. 21.2KiB allocated for chunks. 21.2KiB in use in bin. 2.7KiB client-requested in use in bin.
2024-07-12 14:20:01.744338: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (512): 	Total Chunks: 8, Chunks in use: 8. 4.8KiB allocated for chunks. 4.8KiB in use in bin. 4.0KiB client-requested in use in bin.
2024-07-12 14:20:01.744348: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (1024): 	Total Chunks: 21, Chunks in use: 21. 21.2KiB allocated for chunks. 21.2KiB in use in bin. 21.0KiB client-requested in use in bin.
2024-07-12 14:20:01.744358: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (2048): 	Total Chunks: 24, Chunks in use: 24. 50.0KiB allocated for chunks. 50.0KiB in use in bin. 48.0KiB client-requested in use in bin.
2024-07-12 14:20:01.744367: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (4096): 	Total Chunks: 2, Chunks in use: 2. 13.5KiB allocated for chunks. 13.5KiB in use in bin. 13.5KiB client-requested in use in bin.
2024-07-12 14:20:01.744377: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (8192): 	Total Chunks: 2, Chunks in use: 2. 20.5KiB allocated for chunks. 20.5KiB in use in bin. 13.5KiB client-requested in use in bin.
2024-07-12 14:20:01.744387: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (16384): 	Total Chunks: 10, Chunks in use: 10. 162.0KiB allocated for chunks. 162.0KiB in use in bin. 148.5KiB client-requested in use in bin.
2024-07-12 14:20:01.744406: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (32768): 	Total Chunks: 2, Chunks in use: 2. 66.5KiB allocated for chunks. 66.5KiB in use in bin. 66.4KiB client-requested in use in bin.
2024-07-12 14:20:01.744416: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (65536): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2024-07-12 14:20:01.744426: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (131072): 	Total Chunks: 7, Chunks in use: 7. 1.21MiB allocated for chunks. 1.21MiB in use in bin. 969.5KiB client-requested in use in bin.
2024-07-12 14:20:01.744435: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (262144): 	Total Chunks: 5, Chunks in use: 4. 1.91MiB allocated for chunks. 1.44MiB in use in bin. 1.12MiB client-requested in use in bin.
2024-07-12 14:20:01.744445: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (524288): 	Total Chunks: 50, Chunks in use: 50. 34.90MiB allocated for chunks. 34.90MiB in use in bin. 34.29MiB client-requested in use in bin.
2024-07-12 14:20:01.744455: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (1048576): 	Total Chunks: 4, Chunks in use: 4. 5.34MiB allocated for chunks. 5.34MiB in use in bin. 4.50MiB client-requested in use in bin.
2024-07-12 14:20:01.744465: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (2097152): 	Total Chunks: 8, Chunks in use: 8. 20.69MiB allocated for chunks. 20.69MiB in use in bin. 18.81MiB client-requested in use in bin.
2024-07-12 14:20:01.744474: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (4194304): 	Total Chunks: 8, Chunks in use: 8. 33.98MiB allocated for chunks. 33.98MiB in use in bin. 31.75MiB client-requested in use in bin.
2024-07-12 14:20:01.744484: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (8388608): 	Total Chunks: 25, Chunks in use: 25. 258.56MiB allocated for chunks. 258.56MiB in use in bin. 240.75MiB client-requested in use in bin.
2024-07-12 14:20:01.744494: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (16777216): 	Total Chunks: 3, Chunks in use: 3. 75.81MiB allocated for chunks. 75.81MiB in use in bin. 75.81MiB client-requested in use in bin.
2024-07-12 14:20:01.744504: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (33554432): 	Total Chunks: 4, Chunks in use: 4. 240.25MiB allocated for chunks. 240.25MiB in use in bin. 240.25MiB client-requested in use in bin.
2024-07-12 14:20:01.744513: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (67108864): 	Total Chunks: 9, Chunks in use: 9. 829.75MiB allocated for chunks. 829.75MiB in use in bin. 698.31MiB client-requested in use in bin.
2024-07-12 14:20:01.744523: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (134217728): 	Total Chunks: 3, Chunks in use: 3. 639.21MiB allocated for chunks. 639.21MiB in use in bin. 610.35MiB client-requested in use in bin.
2024-07-12 14:20:01.744533: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (268435456): 	Total Chunks: 14, Chunks in use: 13. 35.38GiB allocated for chunks. 35.02GiB in use in bin. 34.94GiB client-requested in use in bin.
2024-07-12 14:20:01.744543: I tensorflow/tsl/framework/bfc_allocator.cc:1062] Bin for 488.28MiB was 256.00MiB, Chunk State: 
2024-07-12 14:20:01.744561: I tensorflow/tsl/framework/bfc_allocator.cc:1068]   Size: 361.64MiB | Requested Size: 488.28MiB | in_use: 0 | bin_num: 20, prev:   Size: 488.28MiB | Requested Size: 488.28MiB | in_use: 1 | bin_num: -1, next:   Size: 9.00MiB | Requested Size: 9.00MiB | in_use: 1 | bin_num: -1
2024-07-12 14:20:01.744570: I tensorflow/tsl/framework/bfc_allocator.cc:1075] Next region of size 40231108608
2024-07-12 14:20:01.744581: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936000000 of size 256 next 1
2024-07-12 14:20:01.744590: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936000100 of size 1280 next 2
2024-07-12 14:20:01.744598: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936000600 of size 256 next 3
2024-07-12 14:20:01.744608: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936000700 of size 256 next 4
2024-07-12 14:20:01.744617: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936000800 of size 256 next 6
2024-07-12 14:20:01.744625: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936000900 of size 256 next 7
2024-07-12 14:20:01.744632: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936000a00 of size 256 next 5
2024-07-12 14:20:01.744640: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936000b00 of size 256 next 8
2024-07-12 14:20:01.744648: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936000c00 of size 256 next 13
2024-07-12 14:20:01.744656: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936000d00 of size 256 next 11
2024-07-12 14:20:01.744664: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936000e00 of size 256 next 12
2024-07-12 14:20:01.744673: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936000f00 of size 512 next 16
2024-07-12 14:20:01.744683: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936001100 of size 256 next 17
2024-07-12 14:20:01.744691: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936001200 of size 256 next 20
2024-07-12 14:20:01.744699: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936001300 of size 256 next 58
2024-07-12 14:20:01.744707: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936001400 of size 256 next 23
2024-07-12 14:20:01.744715: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936001500 of size 256 next 21
2024-07-12 14:20:01.744723: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936001600 of size 256 next 22
2024-07-12 14:20:01.744731: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936001700 of size 1024 next 26
2024-07-12 14:20:01.744739: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936001b00 of size 256 next 27
2024-07-12 14:20:01.744747: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936001c00 of size 256 next 30
2024-07-12 14:20:01.744755: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936001d00 of size 1024 next 33
2024-07-12 14:20:01.744763: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936002100 of size 1024 next 36
2024-07-12 14:20:01.744770: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936002500 of size 256 next 31
2024-07-12 14:20:01.744778: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936002600 of size 256 next 32
2024-07-12 14:20:01.744786: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936002700 of size 2048 next 38
2024-07-12 14:20:01.744794: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936002f00 of size 256 next 39
2024-07-12 14:20:01.744802: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936003000 of size 256 next 42
2024-07-12 14:20:01.744810: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936003100 of size 2048 next 45
2024-07-12 14:20:01.744818: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936003900 of size 2048 next 9
2024-07-12 14:20:01.744826: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936004100 of size 256 next 66
2024-07-12 14:20:01.744834: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936004200 of size 512 next 18
2024-07-12 14:20:01.744842: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936004400 of size 1024 next 29
2024-07-12 14:20:01.744850: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936004800 of size 2048 next 40
2024-07-12 14:20:01.744858: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936005000 of size 2048 next 50
2024-07-12 14:20:01.744866: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936005800 of size 2048 next 52
2024-07-12 14:20:01.744874: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936006000 of size 256 next 54
2024-07-12 14:20:01.744884: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936006100 of size 256 next 72
2024-07-12 14:20:01.744892: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936006200 of size 256 next 77
2024-07-12 14:20:01.744900: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936006300 of size 256 next 78
2024-07-12 14:20:01.744908: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936006400 of size 256 next 79
2024-07-12 14:20:01.744916: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936006500 of size 256 next 80
2024-07-12 14:20:01.744924: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936006600 of size 256 next 82
2024-07-12 14:20:01.744932: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936006700 of size 512 next 84
2024-07-12 14:20:01.744940: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936006900 of size 768 next 73
2024-07-12 14:20:01.744948: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936006c00 of size 1024 next 74
2024-07-12 14:20:01.744956: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936007000 of size 256 next 70
2024-07-12 14:20:01.744964: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936007100 of size 256 next 71
2024-07-12 14:20:01.745018: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936007200 of size 256 next 68
2024-07-12 14:20:01.745026: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936007300 of size 256 next 53
2024-07-12 14:20:01.745034: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936007400 of size 256 next 48
2024-07-12 14:20:01.745042: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936007500 of size 256 next 43
2024-07-12 14:20:01.745050: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936007600 of size 16384 next 57
2024-07-12 14:20:01.745058: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f593600b600 of size 256 next 55
2024-07-12 14:20:01.745066: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f593600b700 of size 256 next 56
2024-07-12 14:20:01.745074: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f593600b800 of size 16384 next 62
2024-07-12 14:20:01.745082: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f593600f800 of size 256 next 60
2024-07-12 14:20:01.745090: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f593600f900 of size 256 next 61
2024-07-12 14:20:01.745098: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f593600fa00 of size 10496 next 67
2024-07-12 14:20:01.745106: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936012300 of size 6912 next 65
2024-07-12 14:20:01.745115: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936013e00 of size 237056 next 15
2024-07-12 14:20:01.745123: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f593604dc00 of size 442368 next 19
2024-07-12 14:20:01.745131: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59360b9c00 of size 2064384 next 24
2024-07-12 14:20:01.745140: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59362b1c00 of size 589824 next 14
2024-07-12 14:20:01.745148: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936341c00 of size 4718592 next 34
2024-07-12 14:20:01.745156: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59367c1c00 of size 16384 next 59
2024-07-12 14:20:01.745164: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59367c5c00 of size 2359296 next 88
2024-07-12 14:20:01.745172: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936a05c00 of size 4112384 next 37
2024-07-12 14:20:01.745180: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5936df1c00 of size 2359296 next 28
2024-07-12 14:20:01.745188: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5937031c00 of size 2359296 next 41
2024-07-12 14:20:01.745196: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5937271c00 of size 9437184 next 35
2024-07-12 14:20:01.745207: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5937b71c00 of size 9437184 next 47
2024-07-12 14:20:01.745215: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5938471c00 of size 9437184 next 46
2024-07-12 14:20:01.745223: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5938d71c00 of size 9437184 next 10
2024-07-12 14:20:01.745231: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5939671c00 of size 9437184 next 44
2024-07-12 14:20:01.745239: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5939f71c00 of size 109510656 next 69
2024-07-12 14:20:01.745248: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e1c00 of size 1024 next 193
2024-07-12 14:20:01.745256: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e2000 of size 3072 next 148
2024-07-12 14:20:01.745265: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e2c00 of size 2048 next 221
2024-07-12 14:20:01.745273: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e3400 of size 2048 next 199
2024-07-12 14:20:01.745281: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e3c00 of size 2048 next 130
2024-07-12 14:20:01.745289: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e4400 of size 256 next 214
2024-07-12 14:20:01.745297: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e4500 of size 256 next 184
2024-07-12 14:20:01.745305: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e4600 of size 256 next 219
2024-07-12 14:20:01.745313: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e4700 of size 256 next 143
2024-07-12 14:20:01.745321: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e4800 of size 256 next 225
2024-07-12 14:20:01.745329: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e4900 of size 256 next 227
2024-07-12 14:20:01.745337: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e4a00 of size 256 next 228
2024-07-12 14:20:01.745345: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e4b00 of size 256 next 230
2024-07-12 14:20:01.745353: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e4c00 of size 768 next 151
2024-07-12 14:20:01.745361: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e4f00 of size 1024 next 76
2024-07-12 14:20:01.745369: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59407e5300 of size 147456 next 81
2024-07-12 14:20:01.745377: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940809300 of size 294912 next 83
2024-07-12 14:20:01.745385: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940851300 of size 589824 next 85
2024-07-12 14:20:01.745393: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59408e1300 of size 1179648 next 86
2024-07-12 14:20:01.745401: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a01300 of size 1024 next 87
2024-07-12 14:20:01.745409: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a01700 of size 1024 next 89
2024-07-12 14:20:01.745417: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a01b00 of size 1024 next 90
2024-07-12 14:20:01.745425: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a01f00 of size 2048 next 92
2024-07-12 14:20:01.745433: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a02700 of size 2048 next 94
2024-07-12 14:20:01.745441: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a02f00 of size 2048 next 96
2024-07-12 14:20:01.745449: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a03700 of size 2048 next 97
2024-07-12 14:20:01.745457: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a03f00 of size 2048 next 99
2024-07-12 14:20:01.745465: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a04700 of size 2048 next 101
2024-07-12 14:20:01.745475: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a04f00 of size 16384 next 103
2024-07-12 14:20:01.745483: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a08f00 of size 16384 next 105
2024-07-12 14:20:01.745491: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a0cf00 of size 1024 next 107
2024-07-12 14:20:01.745499: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a0d300 of size 1024 next 108
2024-07-12 14:20:01.745507: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a0d700 of size 256 next 109
2024-07-12 14:20:01.745515: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a0d800 of size 256 next 110
2024-07-12 14:20:01.745523: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a0d900 of size 256 next 111
2024-07-12 14:20:01.745531: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a0da00 of size 256 next 112
2024-07-12 14:20:01.745539: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a0db00 of size 256 next 113
2024-07-12 14:20:01.745547: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a0dc00 of size 256 next 114
2024-07-12 14:20:01.745555: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a0dd00 of size 256 next 115
2024-07-12 14:20:01.745563: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a0de00 of size 256 next 116
2024-07-12 14:20:01.745571: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a0df00 of size 256 next 117
2024-07-12 14:20:01.745578: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a0e000 of size 256 next 118
2024-07-12 14:20:01.745586: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a0e100 of size 256 next 119
2024-07-12 14:20:01.745594: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a0e200 of size 256 next 120
2024-07-12 14:20:01.745602: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a0e300 of size 35328 next 204
2024-07-12 14:20:01.745610: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a16d00 of size 256 next 162
2024-07-12 14:20:01.745618: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a16e00 of size 512 next 158
2024-07-12 14:20:01.745626: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a17000 of size 1024 next 152
2024-07-12 14:20:01.745634: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a17400 of size 2048 next 200
2024-07-12 14:20:01.745642: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a17c00 of size 3072 next 161
2024-07-12 14:20:01.745650: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a18800 of size 18432 next 210
2024-07-12 14:20:01.745659: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a1d000 of size 16384 next 201
2024-07-12 14:20:01.745666: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a21000 of size 10496 next 224
2024-07-12 14:20:01.745674: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a23900 of size 6912 next 197
2024-07-12 14:20:01.745682: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a25400 of size 242688 next 145
2024-07-12 14:20:01.745691: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940a60800 of size 477696 next 150
2024-07-12 14:20:01.745699: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad5200 of size 256 next 121
2024-07-12 14:20:01.745707: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad5300 of size 256 next 138
2024-07-12 14:20:01.745715: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad5400 of size 256 next 180
2024-07-12 14:20:01.745723: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad5500 of size 256 next 222
2024-07-12 14:20:01.745730: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad5600 of size 256 next 159
2024-07-12 14:20:01.745738: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad5700 of size 768 next 182
2024-07-12 14:20:01.745749: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad5a00 of size 1024 next 192
2024-07-12 14:20:01.745757: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad5e00 of size 1024 next 140
2024-07-12 14:20:01.745765: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad6200 of size 256 next 135
2024-07-12 14:20:01.745773: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad6300 of size 256 next 133
2024-07-12 14:20:01.745780: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad6400 of size 256 next 123
2024-07-12 14:20:01.745788: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad6500 of size 256 next 156
2024-07-12 14:20:01.745796: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad6600 of size 256 next 154
2024-07-12 14:20:01.745804: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad6700 of size 256 next 122
2024-07-12 14:20:01.745812: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad6800 of size 256 next 127
2024-07-12 14:20:01.745820: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940ad6900 of size 228096 next 124
2024-07-12 14:20:01.745828: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940b0e400 of size 256 next 125
2024-07-12 14:20:01.745836: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940b0e500 of size 866048 next 25
2024-07-12 14:20:01.745845: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940be1c00 of size 4194304 next 49
2024-07-12 14:20:01.745853: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5940fe1c00 of size 4718592 next 91
2024-07-12 14:20:01.745861: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5941461c00 of size 9437184 next 93
2024-07-12 14:20:01.745869: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5941d61c00 of size 9437184 next 95
2024-07-12 14:20:01.745877: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5942661c00 of size 16318464 next 64
2024-07-12 14:20:01.745885: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59435f1c00 of size 67108864 next 63
2024-07-12 14:20:01.745893: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59475f1c00 of size 411041792 next 51
2024-07-12 14:20:01.745902: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f595fdf1c00 of size 256000000 next 270
2024-07-12 14:20:01.745910: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f596f215c00 of size 256000000 next 271
2024-07-12 14:20:01.745918: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f597e639c00 of size 62980096 next 272
2024-07-12 14:20:01.745926: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5982249c00 of size 125960192 next 273
2024-07-12 14:20:01.745934: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5989a69c00 of size 125960192 next 274
2024-07-12 14:20:01.745942: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5991289c00 of size 125960192 next 275
2024-07-12 14:20:01.745950: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5998aa9c00 of size 31490048 next 276
2024-07-12 14:20:01.745958: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f599a8b1c00 of size 62980096 next 277
2024-07-12 14:20:01.745985: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f599e4c1c00 of size 62980096 next 278
2024-07-12 14:20:01.745993: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59a20d1c00 of size 62980096 next 279
2024-07-12 14:20:01.746001: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59a5ce1c00 of size 14745600 next 280
2024-07-12 14:20:01.746010: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59a6af1c00 of size 14745600 next 281
2024-07-12 14:20:01.746018: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59a7901c00 of size 14745600 next 282
2024-07-12 14:20:01.746026: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59a8711c00 of size 14745600 next 283
2024-07-12 14:20:01.746036: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59a9521c00 of size 3211264 next 284
2024-07-12 14:20:01.746044: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59a9831c00 of size 131072 next 287
2024-07-12 14:20:01.746053: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59a9851c00 of size 524288 next 288
2024-07-12 14:20:01.746061: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59a98d1c00 of size 32768 next 291
2024-07-12 14:20:01.746069: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f59a98d9c00 of size 491520 next 289
2024-07-12 14:20:01.746077: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59a9951c00 of size 131072 next 290
2024-07-12 14:20:01.746085: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59a9971c00 of size 512000000 next 292
2024-07-12 14:20:01.746093: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59c81b9c00 of size 512000000 next 293
2024-07-12 14:20:01.746101: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f59e6a01c00 of size 379204608 next 75
2024-07-12 14:20:01.746109: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59fd3a5000 of size 9437184 next 98
2024-07-12 14:20:01.746117: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59fdca5000 of size 9437184 next 100
2024-07-12 14:20:01.746125: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f59fe5a5000 of size 411041792 next 102
2024-07-12 14:20:01.746133: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a16da5000 of size 67108864 next 104
2024-07-12 14:20:01.746141: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1ada5000 of size 4194304 next 106
2024-07-12 14:20:01.746149: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1b1a5000 of size 1179648 next 131
2024-07-12 14:20:01.746157: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1b2c5000 of size 2570752 next 209
2024-07-12 14:20:01.746166: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1b538a00 of size 750080 next 220
2024-07-12 14:20:01.746174: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1b5efc00 of size 750080 next 189
2024-07-12 14:20:01.746182: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1b6a6e00 of size 750080 next 190
2024-07-12 14:20:01.746190: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1b75e000 of size 750080 next 196
2024-07-12 14:20:01.746198: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1b815200 of size 750080 next 126
2024-07-12 14:20:01.746205: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1b8cc400 of size 750080 next 202
2024-07-12 14:20:01.746213: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1b983600 of size 750080 next 155
2024-07-12 14:20:01.746221: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1ba3a800 of size 750080 next 160
2024-07-12 14:20:01.746229: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1baf1a00 of size 750080 next 213
2024-07-12 14:20:01.746237: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1bba8c00 of size 750080 next 137
2024-07-12 14:20:01.746245: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1bc5fe00 of size 750080 next 218
2024-07-12 14:20:01.746253: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1bd17000 of size 750080 next 211
2024-07-12 14:20:01.746261: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1bdce200 of size 750080 next 179
2024-07-12 14:20:01.746269: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1be85400 of size 750080 next 207
2024-07-12 14:20:01.746277: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1bf3c600 of size 750080 next 216
2024-07-12 14:20:01.746284: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1bff3800 of size 750080 next 183
2024-07-12 14:20:01.746292: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1c0aaa00 of size 750080 next 175
2024-07-12 14:20:01.746303: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1c161c00 of size 750080 next 174
2024-07-12 14:20:01.746311: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1c218e00 of size 750080 next 173
2024-07-12 14:20:01.746319: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1c2d0000 of size 750080 next 172
2024-07-12 14:20:01.746327: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1c387200 of size 750080 next 171
2024-07-12 14:20:01.746334: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1c43e400 of size 750080 next 170
2024-07-12 14:20:01.746342: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1c4f5600 of size 750080 next 169
2024-07-12 14:20:01.746350: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1c5ac800 of size 750080 next 168
2024-07-12 14:20:01.746358: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1c663a00 of size 750080 next 167
2024-07-12 14:20:01.746366: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1c71ac00 of size 750080 next 166
2024-07-12 14:20:01.746374: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1c7d1e00 of size 750080 next 165
2024-07-12 14:20:01.746382: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1c889000 of size 750080 next 164
2024-07-12 14:20:01.746390: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1c940200 of size 750080 next 146
2024-07-12 14:20:01.746398: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1c9f7400 of size 750080 next 157
2024-07-12 14:20:01.746406: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1caae600 of size 750080 next 142
2024-07-12 14:20:01.746414: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1cb65800 of size 750080 next 128
2024-07-12 14:20:01.746421: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1cc1ca00 of size 750080 next 177
2024-07-12 14:20:01.746429: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1ccd3c00 of size 750080 next 203
2024-07-12 14:20:01.746437: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1cd8ae00 of size 750080 next 205
2024-07-12 14:20:01.746445: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1ce42000 of size 750080 next 153
2024-07-12 14:20:01.746453: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1cef9200 of size 750080 next 136
2024-07-12 14:20:01.746461: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1cfb0400 of size 750080 next 212
2024-07-12 14:20:01.746469: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1d067600 of size 750080 next 187
2024-07-12 14:20:01.746477: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1d11e800 of size 750080 next 132
2024-07-12 14:20:01.746485: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1d1d5a00 of size 750080 next 198
2024-07-12 14:20:01.746493: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1d28cc00 of size 10616832 next 195
2024-07-12 14:20:01.746501: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1dcacc00 of size 2359296 next 129
2024-07-12 14:20:01.746509: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1deecc00 of size 16384 next 215
2024-07-12 14:20:01.746517: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1def0c00 of size 2359296 next 236
2024-07-12 14:20:01.746525: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1e130c00 of size 4702208 next 186
2024-07-12 14:20:01.746533: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1e5acc00 of size 8449024 next 206
2024-07-12 14:20:01.746541: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1edbb800 of size 750080 next 191
2024-07-12 14:20:01.746549: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a1ee72a00 of size 825000192 next 176
2024-07-12 14:20:01.746558: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a5013ab00 of size 411041792 next 250
2024-07-12 14:20:01.746568: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a6893ab00 of size 512000000 next 269
2024-07-12 14:20:01.746576: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5a87182b00 of size 600734720 next 147
2024-07-12 14:20:01.746584: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5aaae6a700 of size 256 next 149
2024-07-12 14:20:01.746592: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5aaae6a800 of size 13200000000 next 144
2024-07-12 14:20:01.746601: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f5dbdaeac00 of size 13200000000 next 217
2024-07-12 14:20:01.746609: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f60d076b000 of size 3300000000 next 181
2024-07-12 14:20:01.746617: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f619528b100 of size 3300000000 next 139
2024-07-12 14:20:01.746625: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f6259dab200 of size 9437184 next 163
2024-07-12 14:20:01.746633: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f625a6ab200 of size 9437184 next 194
2024-07-12 14:20:01.746641: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f625afab200 of size 9437184 next 185
2024-07-12 14:20:01.746649: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f625b8ab200 of size 9437184 next 134
2024-07-12 14:20:01.746657: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f625c1ab200 of size 114229248 next 226
2024-07-12 14:20:01.746665: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f6262e9b200 of size 147456 next 229
2024-07-12 14:20:01.746673: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f6262ebf200 of size 294912 next 231
2024-07-12 14:20:01.746681: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f6262f07200 of size 589824 next 232
2024-07-12 14:20:01.746689: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f6262f97200 of size 512 next 233
2024-07-12 14:20:01.746697: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f6262f97400 of size 1179648 next 234
2024-07-12 14:20:01.746705: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630b7400 of size 1024 next 235
2024-07-12 14:20:01.746713: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630b7800 of size 1024 next 237
2024-07-12 14:20:01.746721: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630b7c00 of size 1024 next 238
2024-07-12 14:20:01.746729: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630b8000 of size 2048 next 240
2024-07-12 14:20:01.746737: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630b8800 of size 2048 next 242
2024-07-12 14:20:01.746745: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630b9000 of size 2048 next 244
2024-07-12 14:20:01.746752: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630b9800 of size 2048 next 245
2024-07-12 14:20:01.746760: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630ba000 of size 2048 next 247
2024-07-12 14:20:01.746768: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630ba800 of size 2048 next 249
2024-07-12 14:20:01.746776: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630bb000 of size 16384 next 251
2024-07-12 14:20:01.746784: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630bf000 of size 16384 next 253
2024-07-12 14:20:01.746792: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630c3000 of size 1024 next 255
2024-07-12 14:20:01.746800: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630c3400 of size 1024 next 256
2024-07-12 14:20:01.746808: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630c3800 of size 256 next 257
2024-07-12 14:20:01.746816: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630c3900 of size 256 next 258
2024-07-12 14:20:01.746824: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630c3a00 of size 256 next 259
2024-07-12 14:20:01.746834: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630c3b00 of size 256 next 260
2024-07-12 14:20:01.746842: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630c3c00 of size 256 next 261
2024-07-12 14:20:01.746850: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630c3d00 of size 256 next 262
2024-07-12 14:20:01.746858: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630c3e00 of size 256 next 263
2024-07-12 14:20:01.746866: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630c3f00 of size 256 next 264
2024-07-12 14:20:01.746874: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630c4000 of size 256 next 265
2024-07-12 14:20:01.746882: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630c4100 of size 256 next 268
2024-07-12 14:20:01.746890: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62630c4200 of size 524288 next 285
2024-07-12 14:20:01.746898: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f6263144200 of size 524288 next 286
2024-07-12 14:20:01.746906: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62631c4200 of size 880640 next 141
2024-07-12 14:20:01.746914: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f626329b200 of size 4194304 next 208
2024-07-12 14:20:01.746922: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f626369b200 of size 4718592 next 239
2024-07-12 14:20:01.746930: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f6263b1b200 of size 9437184 next 241
2024-07-12 14:20:01.746938: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f626441b200 of size 9437184 next 243
2024-07-12 14:20:01.746946: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f6264d1b200 of size 16318464 next 223
2024-07-12 14:20:01.746954: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f6265cab200 of size 67108864 next 178
2024-07-12 14:20:01.746961: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f6269cab200 of size 411041792 next 188
2024-07-12 14:20:01.746992: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62824ab200 of size 9437184 next 246
2024-07-12 14:20:01.747001: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f6282dab200 of size 9437184 next 248
2024-07-12 14:20:01.747009: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62836ab200 of size 67108864 next 252
2024-07-12 14:20:01.747017: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f62876ab200 of size 4194304 next 254
2024-07-12 14:20:01.747025: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f6287aab200 of size 24000000 next 266
2024-07-12 14:20:01.747033: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f628918e800 of size 24000000 next 267
2024-07-12 14:20:01.747041: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f628a871e00 of size 158261760 next 18446744073709551615
2024-07-12 14:20:01.747049: I tensorflow/tsl/framework/bfc_allocator.cc:1100]      Summary of in-use Chunks by size: 
2024-07-12 14:20:01.747060: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 85 Chunks of size 256 totalling 21.2KiB
2024-07-12 14:20:01.747069: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 5 Chunks of size 512 totalling 2.5KiB
2024-07-12 14:20:01.747077: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 3 Chunks of size 768 totalling 2.2KiB
2024-07-12 14:20:01.747086: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 20 Chunks of size 1024 totalling 20.0KiB
2024-07-12 14:20:01.747095: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 1280 totalling 1.2KiB
2024-07-12 14:20:01.747103: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 22 Chunks of size 2048 totalling 44.0KiB
2024-07-12 14:20:01.747112: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 3072 totalling 6.0KiB
2024-07-12 14:20:01.747120: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 6912 totalling 13.5KiB
2024-07-12 14:20:01.747131: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 10496 totalling 20.5KiB
2024-07-12 14:20:01.747140: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 9 Chunks of size 16384 totalling 144.0KiB
2024-07-12 14:20:01.747149: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 18432 totalling 18.0KiB
2024-07-12 14:20:01.747158: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 32768 totalling 32.0KiB
2024-07-12 14:20:01.747166: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 35328 totalling 34.5KiB
2024-07-12 14:20:01.747175: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 131072 totalling 256.0KiB
2024-07-12 14:20:01.747184: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 147456 totalling 288.0KiB
2024-07-12 14:20:01.747193: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 228096 totalling 222.8KiB
2024-07-12 14:20:01.747201: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 237056 totalling 231.5KiB
2024-07-12 14:20:01.747210: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 242688 totalling 237.0KiB
2024-07-12 14:20:01.747219: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 294912 totalling 576.0KiB
2024-07-12 14:20:01.747227: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 442368 totalling 432.0KiB
2024-07-12 14:20:01.747236: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 477696 totalling 466.5KiB
2024-07-12 14:20:01.747245: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 3 Chunks of size 524288 totalling 1.50MiB
2024-07-12 14:20:01.747253: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 3 Chunks of size 589824 totalling 1.69MiB
2024-07-12 14:20:01.747262: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 42 Chunks of size 750080 totalling 30.04MiB
2024-07-12 14:20:01.747271: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 866048 totalling 845.8KiB
2024-07-12 14:20:01.747279: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 880640 totalling 860.0KiB
2024-07-12 14:20:01.747288: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 3 Chunks of size 1179648 totalling 3.38MiB
2024-07-12 14:20:01.747296: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 2064384 totalling 1.97MiB
2024-07-12 14:20:01.747305: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 5 Chunks of size 2359296 totalling 11.25MiB
2024-07-12 14:20:01.747313: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 2570752 totalling 2.45MiB
2024-07-12 14:20:01.747322: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 3211264 totalling 3.06MiB
2024-07-12 14:20:01.747330: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 4112384 totalling 3.92MiB
2024-07-12 14:20:01.747339: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 4194304 totalling 16.00MiB
2024-07-12 14:20:01.747347: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 4702208 totalling 4.48MiB
2024-07-12 14:20:01.747356: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 3 Chunks of size 4718592 totalling 13.50MiB
2024-07-12 14:20:01.747364: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 8449024 totalling 8.06MiB
2024-07-12 14:20:01.747373: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 17 Chunks of size 9437184 totalling 153.00MiB
2024-07-12 14:20:01.747382: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 10616832 totalling 10.12MiB
2024-07-12 14:20:01.747390: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 14745600 totalling 56.25MiB
2024-07-12 14:20:01.747399: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 16318464 totalling 31.12MiB
2024-07-12 14:20:01.747408: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 24000000 totalling 45.78MiB
2024-07-12 14:20:01.747416: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 31490048 totalling 30.03MiB
2024-07-12 14:20:01.747427: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 62980096 totalling 240.25MiB
2024-07-12 14:20:01.747436: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 67108864 totalling 256.00MiB
2024-07-12 14:20:01.747445: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 109510656 totalling 104.44MiB
2024-07-12 14:20:01.747454: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 114229248 totalling 108.94MiB
2024-07-12 14:20:01.747462: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 3 Chunks of size 125960192 totalling 360.38MiB
2024-07-12 14:20:01.747471: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 158261760 totalling 150.93MiB
2024-07-12 14:20:01.747480: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 256000000 totalling 488.28MiB
2024-07-12 14:20:01.747488: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 411041792 totalling 1.53GiB
2024-07-12 14:20:01.747497: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 3 Chunks of size 512000000 totalling 1.43GiB
2024-07-12 14:20:01.747505: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 600734720 totalling 572.91MiB
2024-07-12 14:20:01.747514: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 825000192 totalling 786.78MiB
2024-07-12 14:20:01.747522: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 3300000000 totalling 6.15GiB
2024-07-12 14:20:01.747531: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 13200000000 totalling 24.59GiB
2024-07-12 14:20:01.747540: I tensorflow/tsl/framework/bfc_allocator.cc:1107] Sum Total of in-use chunks: 37.11GiB
2024-07-12 14:20:01.747548: I tensorflow/tsl/framework/bfc_allocator.cc:1109] Total bytes in pool: 40231108608 memory_limit_: 40231108608 available bytes: 0 curr_region_allocation_bytes_: 80462217216
2024-07-12 14:20:01.747561: I tensorflow/tsl/framework/bfc_allocator.cc:1114] Stats: 
Limit:                     40231108608
InUse:                     39851412480
MaxInUse:                  39875904000
NumAllocs:                     2797476
MaxAllocSize:              13200000000
Reserved:                            0
PeakReserved:                        0
LargestFreeBlock:                    0

2024-07-12 14:20:01.747582: W tensorflow/tsl/framework/bfc_allocator.cc:497] ****************************************************************************************************
2024-07-12 14:20:01.747621: W tensorflow/core/framework/op_kernel.cc:1828] OP_REQUIRES failed at conv_ops_fused_impl.h:452 : RESOURCE_EXHAUSTED: OOM when allocating tensor with shape[32,64,250,250] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
Traceback (most recent call last):
  File "/home/xx4455/Comparable/Code/ImageExp/Experiments.py", line 104, in <module>
    experiment(dataName="FaceImage", col='1', height=250, width=250)
  File "/home/xx4455/Comparable/Code/ImageExp/Experiments.py", line 63, in experiment
    MI_encoder_sex_single) = cl.comparabilityExperiment(
  File "/home/xx4455/Comparable/Code/ImageExp/Classification.py", line 228, in comparabilityExperiment
    dual_encoder = train_model(train=train, val=val, y_true=y_true, shared=True, height=height, width=width)
  File "/home/xx4455/Comparable/Code/ImageExp/Classification.py", line 66, in train_model
    dual_encoder = learn(train, epochs=epochs, validation_data=val, y_true=y_true, shared=shared, height=height,
  File "/home/xx4455/Comparable/Code/ImageExp/Classification.py", line 53, in learn
    dual_encoder.fit(
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/utils/traceback_utils.py", line 70, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/tensorflow/python/eager/execute.py", line 53, in quick_execute
    tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
tensorflow.python.framework.errors_impl.ResourceExhaustedError: Graph execution error:

Detected at node 'dual_encoder_all/model_1/conv2d_17/Relu_1' defined at (most recent call last):
    File "/home/xx4455/Comparable/Code/ImageExp/Experiments.py", line 104, in <module>
      experiment(dataName="FaceImage", col='1', height=250, width=250)
    File "/home/xx4455/Comparable/Code/ImageExp/Experiments.py", line 63, in experiment
      MI_encoder_sex_single) = cl.comparabilityExperiment(
    File "/home/xx4455/Comparable/Code/ImageExp/Classification.py", line 228, in comparabilityExperiment
      dual_encoder = train_model(train=train, val=val, y_true=y_true, shared=True, height=height, width=width)
    File "/home/xx4455/Comparable/Code/ImageExp/Classification.py", line 66, in train_model
      dual_encoder = learn(train, epochs=epochs, validation_data=val, y_true=y_true, shared=shared, height=height,
    File "/home/xx4455/Comparable/Code/ImageExp/Classification.py", line 53, in learn
      dual_encoder.fit(
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/utils/traceback_utils.py", line 65, in error_handler
      return fn(*args, **kwargs)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/engine/training.py", line 1742, in fit
      tmp_logs = self.train_function(iterator)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/engine/training.py", line 1338, in train_function
      return step_function(self, iterator)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/engine/training.py", line 1322, in step_function
      outputs = model.distribute_strategy.run(run_step, args=(data,))
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/engine/training.py", line 1303, in run_step
      outputs = model.train_step(data)
    File "/home/xx4455/Comparable/Code/ImageExp/SharedDualEncoder.py", line 116, in train_step
      encodings_A, encodings_B, y = self(feature, trainable=trainable)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/utils/traceback_utils.py", line 65, in error_handler
      return fn(*args, **kwargs)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/engine/training.py", line 569, in __call__
      return super().__call__(*args, **kwargs)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/utils/traceback_utils.py", line 65, in error_handler
      return fn(*args, **kwargs)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/engine/base_layer.py", line 1150, in __call__
      outputs = call_fn(inputs, *args, **kwargs)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/utils/traceback_utils.py", line 96, in error_handler
      return fn(*args, **kwargs)
    File "/home/xx4455/Comparable/Code/ImageExp/SharedDualEncoder.py", line 94, in call
      encodings_B = self.encoder(features["B"], training=trainable)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/utils/traceback_utils.py", line 65, in error_handler
      return fn(*args, **kwargs)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/engine/training.py", line 569, in __call__
      return super().__call__(*args, **kwargs)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/utils/traceback_utils.py", line 65, in error_handler
      return fn(*args, **kwargs)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/engine/base_layer.py", line 1150, in __call__
      outputs = call_fn(inputs, *args, **kwargs)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/utils/traceback_utils.py", line 96, in error_handler
      return fn(*args, **kwargs)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/engine/functional.py", line 512, in call
      return self._run_internal_graph(inputs, training=training, mask=mask)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/engine/functional.py", line 669, in _run_internal_graph
      outputs = node.layer(*args, **kwargs)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/utils/traceback_utils.py", line 65, in error_handler
      return fn(*args, **kwargs)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/engine/base_layer.py", line 1150, in __call__
      outputs = call_fn(inputs, *args, **kwargs)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/utils/traceback_utils.py", line 96, in error_handler
      return fn(*args, **kwargs)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/layers/convolutional/base_conv.py", line 321, in call
      return self.activation(outputs)
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/activations.py", line 321, in relu
      return backend.relu(
    File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/backend.py", line 5397, in relu
      x = tf.nn.relu(x)
Node: 'dual_encoder_all/model_1/conv2d_17/Relu_1'
OOM when allocating tensor with shape[32,64,250,250] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
	 [[{{node dual_encoder_all/model_1/conv2d_17/Relu_1}}]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info. This isn't available when running in Eager mode.
 [Op:__inference_train_function_1436569]
