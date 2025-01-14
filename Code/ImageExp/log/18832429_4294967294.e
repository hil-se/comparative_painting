[1;31m==>[0m Error: py-keras matches multiple packages.
  Matching packages:
    [0;90mjsj35ex[0m py-keras[0;36m@3.2.1[0m[0;32m%gcc[0m[0;32m@12.3.1[0m[0;35m arch=linux-rhel9-skylake_avx512[0m
    [0;90mukcde6j[0m py-keras[0;36m@3.2.1[0m[0;32m%gcc[0m[0;32m@12.3.1[0m[0;35m arch=linux-rhel9-skylake_avx512[0m
    [0;90m7b6blx5[0m py-keras[0;36m@3.2.1[0m[0;32m%gcc[0m[0;32m@12.3.1[0m[0;35m arch=linux-rhel9-skylake_avx512[0m
  Use a more specific spec (e.g., prepend '/' to the hash).
2024-07-12 14:06:08.775851: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38367 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:3b:00.0, compute capability: 8.0
2024-07-12 14:06:16.094925: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel/dropout/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-12 14:06:16.451700: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:432] Loaded cuDNN version 8700
2024-07-12 14:06:18.140608: I tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:606] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2024-07-12 14:06:18.435382: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f2dc529bf40 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2024-07-12 14:06:18.435425: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB, Compute Capability 8.0
2024-07-12 14:06:18.515955: I ./tensorflow/compiler/jit/device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
2024-07-12 14:40:35.799701: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape indual_encoder_all/model_1/dropout_2/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-12 14:40:36.376488: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 488.42MiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
2024-07-12 14:40:36.376807: W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of memory trying to allocate 244.70MiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
2024-07-12 14:40:46.377091: W tensorflow/tsl/framework/bfc_allocator.cc:485] Allocator (GPU_0_bfc) ran out of memory trying to allocate 120.12MiB (rounded to 125960192)requested by op dual_encoder_all/model_1/conv2d_21/Relu_1
If the cause is memory fragmentation maybe the environment variable 'TF_GPU_ALLOCATOR=cuda_malloc_async' will improve the situation. 
Current allocation summary follows.
Current allocation summary follows.
2024-07-12 14:40:46.377158: I tensorflow/tsl/framework/bfc_allocator.cc:1039] BFCAllocator dump for GPU_0_bfc
2024-07-12 14:40:46.377174: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (256): 	Total Chunks: 85, Chunks in use: 85. 21.2KiB allocated for chunks. 21.2KiB in use in bin. 2.7KiB client-requested in use in bin.
2024-07-12 14:40:46.377185: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (512): 	Total Chunks: 8, Chunks in use: 8. 4.5KiB allocated for chunks. 4.5KiB in use in bin. 4.0KiB client-requested in use in bin.
2024-07-12 14:40:46.377195: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (1024): 	Total Chunks: 21, Chunks in use: 21. 22.0KiB allocated for chunks. 22.0KiB in use in bin. 21.0KiB client-requested in use in bin.
2024-07-12 14:40:46.377219: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (2048): 	Total Chunks: 24, Chunks in use: 24. 50.0KiB allocated for chunks. 50.0KiB in use in bin. 48.0KiB client-requested in use in bin.
2024-07-12 14:40:46.377237: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (4096): 	Total Chunks: 1, Chunks in use: 1. 6.8KiB allocated for chunks. 6.8KiB in use in bin. 6.8KiB client-requested in use in bin.
2024-07-12 14:40:46.377248: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (8192): 	Total Chunks: 3, Chunks in use: 3. 30.0KiB allocated for chunks. 30.0KiB in use in bin. 20.2KiB client-requested in use in bin.
2024-07-12 14:40:46.377259: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (16384): 	Total Chunks: 10, Chunks in use: 10. 160.0KiB allocated for chunks. 160.0KiB in use in bin. 148.5KiB client-requested in use in bin.
2024-07-12 14:40:46.377269: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (32768): 	Total Chunks: 2, Chunks in use: 2. 66.5KiB allocated for chunks. 66.5KiB in use in bin. 66.4KiB client-requested in use in bin.
2024-07-12 14:40:46.377279: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (65536): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2024-07-12 14:40:46.377289: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (131072): 	Total Chunks: 7, Chunks in use: 7. 1.13MiB allocated for chunks. 1.13MiB in use in bin. 969.5KiB client-requested in use in bin.
2024-07-12 14:40:46.377299: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (262144): 	Total Chunks: 5, Chunks in use: 4. 1.88MiB allocated for chunks. 1.41MiB in use in bin. 1.12MiB client-requested in use in bin.
2024-07-12 14:40:46.377309: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (524288): 	Total Chunks: 52, Chunks in use: 52. 36.40MiB allocated for chunks. 36.40MiB in use in bin. 35.87MiB client-requested in use in bin.
2024-07-12 14:40:46.377319: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (1048576): 	Total Chunks: 5, Chunks in use: 5. 7.21MiB allocated for chunks. 7.21MiB in use in bin. 5.06MiB client-requested in use in bin.
2024-07-12 14:40:46.377329: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (2097152): 	Total Chunks: 10, Chunks in use: 10. 27.00MiB allocated for chunks. 27.00MiB in use in bin. 23.31MiB client-requested in use in bin.
2024-07-12 14:40:46.377339: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (4194304): 	Total Chunks: 8, Chunks in use: 8. 34.00MiB allocated for chunks. 34.00MiB in use in bin. 34.00MiB client-requested in use in bin.
2024-07-12 14:40:46.377350: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (8388608): 	Total Chunks: 24, Chunks in use: 24. 249.38MiB allocated for chunks. 249.38MiB in use in bin. 236.25MiB client-requested in use in bin.
2024-07-12 14:40:46.377360: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (16777216): 	Total Chunks: 3, Chunks in use: 3. 75.81MiB allocated for chunks. 75.81MiB in use in bin. 75.81MiB client-requested in use in bin.
2024-07-12 14:40:46.377371: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (33554432): 	Total Chunks: 5, Chunks in use: 5. 300.31MiB allocated for chunks. 300.31MiB in use in bin. 300.31MiB client-requested in use in bin.
2024-07-12 14:40:46.377380: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (67108864): 	Total Chunks: 13, Chunks in use: 12. 1.28GiB allocated for chunks. 1.16GiB in use in bin. 1.04GiB client-requested in use in bin.
2024-07-12 14:40:46.377390: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (134217728): 	Total Chunks: 6, Chunks in use: 6. 1.30GiB allocated for chunks. 1.30GiB in use in bin. 1.19GiB client-requested in use in bin.
2024-07-12 14:40:46.377400: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (268435456): 	Total Chunks: 12, Chunks in use: 12. 34.17GiB allocated for chunks. 34.17GiB in use in bin. 34.17GiB client-requested in use in bin.
2024-07-12 14:40:46.377410: I tensorflow/tsl/framework/bfc_allocator.cc:1062] Bin for 120.12MiB was 64.00MiB, Chunk State: 
2024-07-12 14:40:46.377427: I tensorflow/tsl/framework/bfc_allocator.cc:1068]   Size: 118.75MiB | Requested Size: 61.19MiB | in_use: 0 | bin_num: 18, prev:   Size: 2.25MiB | Requested Size: 2.25MiB | in_use: 1 | bin_num: -1, next:   Size: 9.00MiB | Requested Size: 9.00MiB | in_use: 1 | bin_num: -1
2024-07-12 14:40:46.377439: I tensorflow/tsl/framework/bfc_allocator.cc:1075] Next region of size 40231108608
2024-07-12 14:40:46.377450: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72000000 of size 256 next 1
2024-07-12 14:40:46.377459: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72000100 of size 1280 next 2
2024-07-12 14:40:46.377468: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72000600 of size 256 next 3
2024-07-12 14:40:46.377477: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72000700 of size 256 next 4
2024-07-12 14:40:46.377485: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72000800 of size 256 next 6
2024-07-12 14:40:46.377494: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72000900 of size 256 next 7
2024-07-12 14:40:46.377502: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72000a00 of size 256 next 5
2024-07-12 14:40:46.377511: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72000b00 of size 256 next 8
2024-07-12 14:40:46.377519: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72000c00 of size 256 next 13
2024-07-12 14:40:46.377528: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72000d00 of size 256 next 11
2024-07-12 14:40:46.377537: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72000e00 of size 256 next 12
2024-07-12 14:40:46.377545: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72000f00 of size 512 next 16
2024-07-12 14:40:46.377555: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72001100 of size 256 next 17
2024-07-12 14:40:46.377564: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72001200 of size 256 next 20
2024-07-12 14:40:46.377572: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72001300 of size 256 next 58
2024-07-12 14:40:46.377581: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72001400 of size 256 next 23
2024-07-12 14:40:46.377589: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72001500 of size 256 next 21
2024-07-12 14:40:46.377598: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72001600 of size 256 next 22
2024-07-12 14:40:46.377607: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72001700 of size 1024 next 26
2024-07-12 14:40:46.377615: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72001b00 of size 256 next 27
2024-07-12 14:40:46.377624: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72001c00 of size 256 next 30
2024-07-12 14:40:46.377632: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72001d00 of size 1024 next 33
2024-07-12 14:40:46.377641: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72002100 of size 1024 next 36
2024-07-12 14:40:46.377649: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72002500 of size 256 next 31
2024-07-12 14:40:46.377658: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72002600 of size 256 next 32
2024-07-12 14:40:46.377667: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72002700 of size 2048 next 38
2024-07-12 14:40:46.377675: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72002f00 of size 256 next 39
2024-07-12 14:40:46.377684: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72003000 of size 256 next 42
2024-07-12 14:40:46.377692: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72003100 of size 2048 next 45
2024-07-12 14:40:46.377701: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72003900 of size 2048 next 9
2024-07-12 14:40:46.377709: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72004100 of size 256 next 66
2024-07-12 14:40:46.377718: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72004200 of size 512 next 18
2024-07-12 14:40:46.377729: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72004400 of size 1024 next 29
2024-07-12 14:40:46.377738: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72004800 of size 2048 next 40
2024-07-12 14:40:46.377746: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72005000 of size 2048 next 50
2024-07-12 14:40:46.377755: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72005800 of size 2048 next 52
2024-07-12 14:40:46.377763: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72006000 of size 256 next 54
2024-07-12 14:40:46.377772: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72006100 of size 256 next 72
2024-07-12 14:40:46.377781: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72006200 of size 256 next 77
2024-07-12 14:40:46.377789: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72006300 of size 256 next 78
2024-07-12 14:40:46.377798: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72006400 of size 256 next 79
2024-07-12 14:40:46.377806: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72006500 of size 256 next 80
2024-07-12 14:40:46.377815: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72006600 of size 256 next 82
2024-07-12 14:40:46.377824: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72006700 of size 512 next 84
2024-07-12 14:40:46.377832: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72006900 of size 768 next 73
2024-07-12 14:40:46.377842: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72006c00 of size 1024 next 74
2024-07-12 14:40:46.377850: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72007000 of size 256 next 70
2024-07-12 14:40:46.377859: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72007100 of size 256 next 71
2024-07-12 14:40:46.377867: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72007200 of size 256 next 68
2024-07-12 14:40:46.377876: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72007300 of size 256 next 53
2024-07-12 14:40:46.377885: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72007400 of size 256 next 48
2024-07-12 14:40:46.377893: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72007500 of size 256 next 43
2024-07-12 14:40:46.377902: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72007600 of size 16384 next 57
2024-07-12 14:40:46.377910: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7200b600 of size 256 next 55
2024-07-12 14:40:46.377919: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7200b700 of size 256 next 56
2024-07-12 14:40:46.377928: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7200b800 of size 16384 next 62
2024-07-12 14:40:46.377936: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7200f800 of size 256 next 60
2024-07-12 14:40:46.377945: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7200f900 of size 256 next 61
2024-07-12 14:40:46.377953: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7200fa00 of size 10496 next 67
2024-07-12 14:40:46.377962: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72012300 of size 6912 next 65
2024-07-12 14:40:46.377971: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72013e00 of size 237056 next 15
2024-07-12 14:40:46.377980: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7204dc00 of size 442368 next 19
2024-07-12 14:40:46.377989: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e720b9c00 of size 2064384 next 24
2024-07-12 14:40:46.377998: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e722b1c00 of size 589824 next 14
2024-07-12 14:40:46.378007: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72341c00 of size 4718592 next 34
2024-07-12 14:40:46.378016: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e727c1c00 of size 16384 next 59
2024-07-12 14:40:46.378028: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e727c5c00 of size 2359296 next 88
2024-07-12 14:40:46.378037: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72a05c00 of size 4112384 next 37
2024-07-12 14:40:46.378047: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e72df1c00 of size 2359296 next 28
2024-07-12 14:40:46.378056: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e73031c00 of size 2359296 next 41
2024-07-12 14:40:46.378065: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e73271c00 of size 9437184 next 35
2024-07-12 14:40:46.378074: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e73b71c00 of size 9437184 next 47
2024-07-12 14:40:46.378082: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e74471c00 of size 9437184 next 46
2024-07-12 14:40:46.378091: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e74d71c00 of size 9437184 next 10
2024-07-12 14:40:46.378100: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e75671c00 of size 9437184 next 44
2024-07-12 14:40:46.378108: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e75f71c00 of size 109510656 next 69
2024-07-12 14:40:46.378117: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e1c00 of size 2048 next 214
2024-07-12 14:40:46.378126: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e2400 of size 1024 next 209
2024-07-12 14:40:46.378135: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e2800 of size 256 next 141
2024-07-12 14:40:46.378143: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e2900 of size 256 next 213
2024-07-12 14:40:46.378152: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e2a00 of size 256 next 204
2024-07-12 14:40:46.378160: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e2b00 of size 256 next 212
2024-07-12 14:40:46.378169: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e2c00 of size 9728 next 201
2024-07-12 14:40:46.378178: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e5200 of size 256 next 185
2024-07-12 14:40:46.378187: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e5300 of size 512 next 206
2024-07-12 14:40:46.378195: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e5500 of size 1024 next 147
2024-07-12 14:40:46.378217: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e5900 of size 2048 next 150
2024-07-12 14:40:46.378227: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e6100 of size 3072 next 200
2024-07-12 14:40:46.378235: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e6d00 of size 2048 next 184
2024-07-12 14:40:46.378244: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e7500 of size 2048 next 216
2024-07-12 14:40:46.378253: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e7d00 of size 256 next 228
2024-07-12 14:40:46.378261: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e7e00 of size 256 next 231
2024-07-12 14:40:46.378270: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e7f00 of size 256 next 232
2024-07-12 14:40:46.378278: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e8000 of size 256 next 234
2024-07-12 14:40:46.378287: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e8100 of size 512 next 236
2024-07-12 14:40:46.378296: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e8300 of size 512 next 230
2024-07-12 14:40:46.378304: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e8500 of size 1280 next 76
2024-07-12 14:40:46.378313: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c7e8a00 of size 147456 next 81
2024-07-12 14:40:46.378322: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c80ca00 of size 294912 next 83
2024-07-12 14:40:46.378331: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c854a00 of size 589824 next 85
2024-07-12 14:40:46.378342: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7c8e4a00 of size 1179648 next 86
2024-07-12 14:40:46.378351: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca04a00 of size 1024 next 87
2024-07-12 14:40:46.378360: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca04e00 of size 1024 next 89
2024-07-12 14:40:46.378368: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca05200 of size 1024 next 90
2024-07-12 14:40:46.378377: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca05600 of size 2048 next 92
2024-07-12 14:40:46.378385: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca05e00 of size 2048 next 94
2024-07-12 14:40:46.378394: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca06600 of size 2048 next 96
2024-07-12 14:40:46.378403: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca06e00 of size 2048 next 97
2024-07-12 14:40:46.378411: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca07600 of size 2048 next 99
2024-07-12 14:40:46.378420: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca07e00 of size 2048 next 101
2024-07-12 14:40:46.378428: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca08600 of size 16384 next 103
2024-07-12 14:40:46.378437: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca0c600 of size 16384 next 105
2024-07-12 14:40:46.378446: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca10600 of size 1024 next 107
2024-07-12 14:40:46.378454: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca10a00 of size 1024 next 108
2024-07-12 14:40:46.378463: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca10e00 of size 256 next 109
2024-07-12 14:40:46.378471: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca10f00 of size 256 next 110
2024-07-12 14:40:46.378480: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca11000 of size 256 next 111
2024-07-12 14:40:46.378489: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca11100 of size 256 next 112
2024-07-12 14:40:46.378497: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca11200 of size 256 next 113
2024-07-12 14:40:46.378506: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca11300 of size 256 next 114
2024-07-12 14:40:46.378514: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca11400 of size 256 next 115
2024-07-12 14:40:46.378523: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca11500 of size 256 next 116
2024-07-12 14:40:46.378532: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca11600 of size 256 next 117
2024-07-12 14:40:46.378540: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca11700 of size 256 next 118
2024-07-12 14:40:46.378549: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca11800 of size 256 next 119
2024-07-12 14:40:46.378557: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca11900 of size 256 next 120
2024-07-12 14:40:46.378566: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7ca11a00 of size 1048832 next 124
2024-07-12 14:40:46.378575: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb11b00 of size 256 next 125
2024-07-12 14:40:46.378583: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb11c00 of size 256 next 139
2024-07-12 14:40:46.378592: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb11d00 of size 256 next 137
2024-07-12 14:40:46.378600: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb11e00 of size 256 next 138
2024-07-12 14:40:46.378609: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb11f00 of size 256 next 161
2024-07-12 14:40:46.378618: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb12000 of size 256 next 196
2024-07-12 14:40:46.378629: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb12100 of size 256 next 199
2024-07-12 14:40:46.378637: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb12200 of size 768 next 197
2024-07-12 14:40:46.378646: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb12500 of size 1536 next 198
2024-07-12 14:40:46.378655: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb12b00 of size 1024 next 192
2024-07-12 14:40:46.378664: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb12f00 of size 3072 next 136
2024-07-12 14:40:46.378672: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb13b00 of size 256 next 130
2024-07-12 14:40:46.378681: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb13c00 of size 256 next 156
2024-07-12 14:40:46.378689: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb13d00 of size 256 next 121
2024-07-12 14:40:46.378698: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb13e00 of size 256 next 142
2024-07-12 14:40:46.378707: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb13f00 of size 256 next 132
2024-07-12 14:40:46.378715: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb14000 of size 256 next 128
2024-07-12 14:40:46.378724: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cb14100 of size 842496 next 25
2024-07-12 14:40:46.378733: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cbe1c00 of size 4194304 next 49
2024-07-12 14:40:46.378742: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7cfe1c00 of size 4718592 next 91
2024-07-12 14:40:46.378750: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7d461c00 of size 9437184 next 93
2024-07-12 14:40:46.378759: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7dd61c00 of size 9437184 next 95
2024-07-12 14:40:46.378768: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7e661c00 of size 16318464 next 64
2024-07-12 14:40:46.378776: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e7f5f1c00 of size 67108864 next 63
2024-07-12 14:40:46.378785: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e835f1c00 of size 411041792 next 51
2024-07-12 14:40:46.378794: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2e9bdf1c00 of size 125960192 next 277
2024-07-12 14:40:46.378803: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2ea3611c00 of size 31490048 next 278
2024-07-12 14:40:46.378812: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2ea5419c00 of size 62980096 next 279
2024-07-12 14:40:46.378821: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2ea9029c00 of size 62980096 next 280
2024-07-12 14:40:46.378829: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2eacc39c00 of size 62980096 next 281
2024-07-12 14:40:46.378838: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2eb0849c00 of size 14745600 next 282
2024-07-12 14:40:46.378847: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2eb1659c00 of size 14745600 next 283
2024-07-12 14:40:46.378856: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2eb2469c00 of size 14745600 next 284
2024-07-12 14:40:46.378864: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2eb3279c00 of size 14745600 next 285
2024-07-12 14:40:46.378873: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2eb4089c00 of size 524288 next 286
2024-07-12 14:40:46.378882: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2eb4109c00 of size 524288 next 287
2024-07-12 14:40:46.378890: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2eb4189c00 of size 524288 next 288
2024-07-12 14:40:46.378899: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2eb4209c00 of size 131072 next 289
2024-07-12 14:40:46.378908: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2eb4229c00 of size 524288 next 290
2024-07-12 14:40:46.378919: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2eb42a9c00 of size 32768 next 293
2024-07-12 14:40:46.378928: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f2eb42b1c00 of size 491520 next 291
2024-07-12 14:40:46.378937: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2eb4329c00 of size 131072 next 292
2024-07-12 14:40:46.378945: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2eb4349c00 of size 512000000 next 294
2024-07-12 14:40:46.378954: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2ed2b91c00 of size 512000000 next 295
2024-07-12 14:40:46.378963: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2ef13d9c00 of size 128000000 next 296
2024-07-12 14:40:46.378972: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2ef8debc00 of size 256000000 next 297
2024-07-12 14:40:46.378981: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f0820fc00 of size 256000000 next 298
2024-07-12 14:40:46.378989: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f17633c00 of size 62980096 next 299
2024-07-12 14:40:46.378998: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f1b243c00 of size 125960192 next 300
2024-07-12 14:40:46.379006: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f22a63c00 of size 125960192 next 301
2024-07-12 14:40:46.379015: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f2a283c00 of size 125960192 next 302
2024-07-12 14:40:46.379024: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f31aa3c00 of size 2359296 next 303
2024-07-12 14:40:46.379032: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f2f31ce3c00 of size 124523520 next 75
2024-07-12 14:40:46.379041: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f393a5000 of size 9437184 next 98
2024-07-12 14:40:46.379050: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f39ca5000 of size 9437184 next 100
2024-07-12 14:40:46.379058: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f3a5a5000 of size 411041792 next 102
2024-07-12 14:40:46.379067: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f52da5000 of size 67108864 next 104
2024-07-12 14:40:46.379076: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f56da5000 of size 4194304 next 106
2024-07-12 14:40:46.379084: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f571a5000 of size 16384 next 223
2024-07-12 14:40:46.379093: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f571a9000 of size 147456 next 233
2024-07-12 14:40:46.379102: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f571cd000 of size 294912 next 235
2024-07-12 14:40:46.379110: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57215000 of size 1041408 next 148
2024-07-12 14:40:46.379119: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57313400 of size 140800 next 153
2024-07-12 14:40:46.379128: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57335a00 of size 35328 next 127
2024-07-12 14:40:46.379137: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f5733e400 of size 16384 next 222
2024-07-12 14:40:46.379146: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57342400 of size 16384 next 224
2024-07-12 14:40:46.379154: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57346400 of size 10496 next 227
2024-07-12 14:40:46.379163: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57348d00 of size 251648 next 188
2024-07-12 14:40:46.379172: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57386400 of size 442368 next 203
2024-07-12 14:40:46.379181: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f573f2400 of size 2086912 next 178
2024-07-12 14:40:46.379190: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f575efc00 of size 750080 next 191
2024-07-12 14:40:46.379198: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f576a6e00 of size 750080 next 183
2024-07-12 14:40:46.379222: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f5775e000 of size 750080 next 160
2024-07-12 14:40:46.379231: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57815200 of size 750080 next 194
2024-07-12 14:40:46.379240: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f578cc400 of size 750080 next 186
2024-07-12 14:40:46.379249: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57983600 of size 750080 next 218
2024-07-12 14:40:46.379257: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57a3a800 of size 750080 next 193
2024-07-12 14:40:46.379266: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57af1a00 of size 750080 next 211
2024-07-12 14:40:46.379275: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57ba8c00 of size 750080 next 195
2024-07-12 14:40:46.379283: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57c5fe00 of size 750080 next 129
2024-07-12 14:40:46.379292: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57d17000 of size 750080 next 173
2024-07-12 14:40:46.379301: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57dce200 of size 750080 next 176
2024-07-12 14:40:46.379309: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57e85400 of size 750080 next 171
2024-07-12 14:40:46.379318: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57f3c600 of size 750080 next 174
2024-07-12 14:40:46.379327: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f57ff3800 of size 750080 next 155
2024-07-12 14:40:46.379335: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f580aaa00 of size 750080 next 169
2024-07-12 14:40:46.379344: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f58161c00 of size 750080 next 168
2024-07-12 14:40:46.379353: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f58218e00 of size 750080 next 167
2024-07-12 14:40:46.379361: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f582d0000 of size 750080 next 166
2024-07-12 14:40:46.379370: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f58387200 of size 750080 next 165
2024-07-12 14:40:46.379379: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f5843e400 of size 750080 next 164
2024-07-12 14:40:46.379387: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f584f5600 of size 750080 next 163
2024-07-12 14:40:46.379396: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f585ac800 of size 750080 next 135
2024-07-12 14:40:46.379404: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f58663a00 of size 750080 next 157
2024-07-12 14:40:46.379413: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f5871ac00 of size 750080 next 143
2024-07-12 14:40:46.379422: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f587d1e00 of size 750080 next 123
2024-07-12 14:40:46.379430: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f58889000 of size 750080 next 158
2024-07-12 14:40:46.379439: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f58940200 of size 750080 next 159
2024-07-12 14:40:46.379448: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f589f7400 of size 750080 next 162
2024-07-12 14:40:46.379456: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f58aae600 of size 750080 next 175
2024-07-12 14:40:46.379465: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f58b65800 of size 750080 next 177
2024-07-12 14:40:46.379474: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f58c1ca00 of size 750080 next 170
2024-07-12 14:40:46.379482: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f58cd3c00 of size 750080 next 180
2024-07-12 14:40:46.379491: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f58d8ae00 of size 750080 next 179
2024-07-12 14:40:46.379500: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f58e42000 of size 750080 next 149
2024-07-12 14:40:46.379510: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f58ef9200 of size 750080 next 140
2024-07-12 14:40:46.379519: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f58fb0400 of size 750080 next 219
2024-07-12 14:40:46.379528: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f59067600 of size 750080 next 215
2024-07-12 14:40:46.379536: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f5911e800 of size 750080 next 202
2024-07-12 14:40:46.379545: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f591d5a00 of size 750080 next 189
2024-07-12 14:40:46.379554: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f5928cc00 of size 750080 next 207
2024-07-12 14:40:46.379562: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f59343e00 of size 750080 next 122
2024-07-12 14:40:46.379571: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f593fb000 of size 750080 next 145
2024-07-12 14:40:46.379580: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f594b2200 of size 750080 next 144
2024-07-12 14:40:46.379588: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f59569400 of size 411041792 next 252
2024-07-12 14:40:46.379597: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f71d69400 of size 512000000 next 271
2024-07-12 14:40:46.379606: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2f905b1400 of size 512000000 next 272
2024-07-12 14:40:46.379614: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2faedf9400 of size 256000000 next 273
2024-07-12 14:40:46.379623: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2fbe21d400 of size 256000000 next 274
2024-07-12 14:40:46.379632: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2fcd641400 of size 62980096 next 275
2024-07-12 14:40:46.379641: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2fd1251400 of size 125960192 next 276
2024-07-12 14:40:46.379650: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2fd8a71400 of size 238522880 next 146
2024-07-12 14:40:46.379659: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2fe6dea600 of size 256 next 151
2024-07-12 14:40:46.379667: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f2fe6dea700 of size 13200000000 next 182
2024-07-12 14:40:46.379676: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f32f9a6ab00 of size 13200000000 next 220
2024-07-12 14:40:46.379685: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f360c6eaf00 of size 3300000000 next 126
2024-07-12 14:40:46.379694: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f36d120b000 of size 3300000000 next 131
2024-07-12 14:40:46.379703: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3795d2b100 of size 1179648 next 181
2024-07-12 14:40:46.379711: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3795e4b100 of size 4718592 next 210
2024-07-12 14:40:46.379720: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37962cb100 of size 1024 next 238
2024-07-12 14:40:46.379729: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37962cb500 of size 2359296 next 239
2024-07-12 14:40:46.379738: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f379650b500 of size 1024 next 240
2024-07-12 14:40:46.379746: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f379650b900 of size 2048 next 242
2024-07-12 14:40:46.379755: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f379650c100 of size 2048 next 244
2024-07-12 14:40:46.379764: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f379650c900 of size 2048 next 246
2024-07-12 14:40:46.379772: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f379650d100 of size 2048 next 247
2024-07-12 14:40:46.379781: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f379650d900 of size 2048 next 249
2024-07-12 14:40:46.379789: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f379650e100 of size 2048 next 251
2024-07-12 14:40:46.379800: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f379650e900 of size 16384 next 253
2024-07-12 14:40:46.379809: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796512900 of size 16384 next 255
2024-07-12 14:40:46.379818: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796516900 of size 1024 next 257
2024-07-12 14:40:46.379826: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796516d00 of size 1024 next 258
2024-07-12 14:40:46.379835: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796517100 of size 256 next 259
2024-07-12 14:40:46.379843: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796517200 of size 256 next 260
2024-07-12 14:40:46.379852: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796517300 of size 256 next 261
2024-07-12 14:40:46.379861: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796517400 of size 256 next 262
2024-07-12 14:40:46.379869: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796517500 of size 256 next 263
2024-07-12 14:40:46.379878: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796517600 of size 256 next 264
2024-07-12 14:40:46.379886: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796517700 of size 256 next 265
2024-07-12 14:40:46.379895: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796517800 of size 256 next 266
2024-07-12 14:40:46.379904: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796517900 of size 256 next 267
2024-07-12 14:40:46.379912: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796517a00 of size 256 next 270
2024-07-12 14:40:46.379921: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796517b00 of size 3487232 next 217
2024-07-12 14:40:46.379930: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f379686b100 of size 2359296 next 205
2024-07-12 14:40:46.379938: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796aab100 of size 2359296 next 208
2024-07-12 14:40:46.379947: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3796ceb100 of size 9437184 next 134
2024-07-12 14:40:46.379956: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37975eb100 of size 9437184 next 154
2024-07-12 14:40:46.379964: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f3797eeb100 of size 9437184 next 172
2024-07-12 14:40:46.379973: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37987eb100 of size 9437184 next 221
2024-07-12 14:40:46.379982: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37990eb100 of size 9437184 next 133
2024-07-12 14:40:46.379990: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37999eb100 of size 109510656 next 229
2024-07-12 14:40:46.379999: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37a025b100 of size 1024 next 237
2024-07-12 14:40:46.380008: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37a025b500 of size 4193280 next 152
2024-07-12 14:40:46.380017: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37a065b100 of size 4194304 next 190
2024-07-12 14:40:46.380025: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37a0a5b100 of size 4718592 next 241
2024-07-12 14:40:46.380034: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37a0edb100 of size 9437184 next 243
2024-07-12 14:40:46.380043: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37a17db100 of size 9437184 next 245
2024-07-12 14:40:46.380051: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37a20db100 of size 16318464 next 226
2024-07-12 14:40:46.380060: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37a306b100 of size 67108864 next 225
2024-07-12 14:40:46.380069: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37a706b100 of size 411041792 next 187
2024-07-12 14:40:46.380077: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37bf86b100 of size 9437184 next 248
2024-07-12 14:40:46.380088: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37c016b100 of size 9437184 next 250
2024-07-12 14:40:46.380097: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37c0a6b100 of size 67108864 next 254
2024-07-12 14:40:46.380105: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37c4a6b100 of size 4194304 next 256
2024-07-12 14:40:46.380114: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37c4e6b100 of size 24000000 next 268
2024-07-12 14:40:46.380123: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37c654e700 of size 24000000 next 269
2024-07-12 14:40:46.380132: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f37c7c31d00 of size 137552640 next 18446744073709551615
2024-07-12 14:40:46.380141: I tensorflow/tsl/framework/bfc_allocator.cc:1100]      Summary of in-use Chunks by size: 
2024-07-12 14:40:46.380152: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 85 Chunks of size 256 totalling 21.2KiB
2024-07-12 14:40:46.380161: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 6 Chunks of size 512 totalling 3.0KiB
2024-07-12 14:40:46.380170: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 768 totalling 1.5KiB
2024-07-12 14:40:46.380179: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 18 Chunks of size 1024 totalling 18.0KiB
2024-07-12 14:40:46.380188: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 1280 totalling 2.5KiB
2024-07-12 14:40:46.380197: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 1536 totalling 1.5KiB
2024-07-12 14:40:46.380219: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 22 Chunks of size 2048 totalling 44.0KiB
2024-07-12 14:40:46.380228: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 3072 totalling 6.0KiB
2024-07-12 14:40:46.380237: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 6912 totalling 6.8KiB
2024-07-12 14:40:46.380246: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 9728 totalling 9.5KiB
2024-07-12 14:40:46.380255: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 10496 totalling 20.5KiB
2024-07-12 14:40:46.380264: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 10 Chunks of size 16384 totalling 160.0KiB
2024-07-12 14:40:46.380273: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 32768 totalling 32.0KiB
2024-07-12 14:40:46.380282: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 35328 totalling 34.5KiB
2024-07-12 14:40:46.380291: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 131072 totalling 256.0KiB
2024-07-12 14:40:46.380300: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 140800 totalling 137.5KiB
2024-07-12 14:40:46.380309: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 147456 totalling 288.0KiB
2024-07-12 14:40:46.380318: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 237056 totalling 231.5KiB
2024-07-12 14:40:46.380327: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 251648 totalling 245.8KiB
2024-07-12 14:40:46.380336: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 294912 totalling 576.0KiB
2024-07-12 14:40:46.380345: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 442368 totalling 864.0KiB
2024-07-12 14:40:46.380354: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 524288 totalling 2.00MiB
2024-07-12 14:40:46.380363: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 589824 totalling 1.12MiB
2024-07-12 14:40:46.380372: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 44 Chunks of size 750080 totalling 31.47MiB
2024-07-12 14:40:46.380381: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 842496 totalling 822.8KiB
2024-07-12 14:40:46.380390: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 1041408 totalling 1017.0KiB
2024-07-12 14:40:46.380399: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 1048832 totalling 1.00MiB
2024-07-12 14:40:46.380410: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 1179648 totalling 2.25MiB
2024-07-12 14:40:46.380419: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 2064384 totalling 1.97MiB
2024-07-12 14:40:46.380428: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 2086912 totalling 1.99MiB
2024-07-12 14:40:46.380437: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 7 Chunks of size 2359296 totalling 15.75MiB
2024-07-12 14:40:46.380446: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 3487232 totalling 3.33MiB
2024-07-12 14:40:46.380455: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 4112384 totalling 3.92MiB
2024-07-12 14:40:46.380464: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 4193280 totalling 4.00MiB
2024-07-12 14:40:46.380473: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 4194304 totalling 16.00MiB
2024-07-12 14:40:46.380482: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 4718592 totalling 18.00MiB
2024-07-12 14:40:46.380491: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 18 Chunks of size 9437184 totalling 162.00MiB
2024-07-12 14:40:46.380500: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 14745600 totalling 56.25MiB
2024-07-12 14:40:46.380509: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 16318464 totalling 31.12MiB
2024-07-12 14:40:46.380518: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 24000000 totalling 45.78MiB
2024-07-12 14:40:46.380527: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 31490048 totalling 30.03MiB
2024-07-12 14:40:46.380536: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 5 Chunks of size 62980096 totalling 300.31MiB
2024-07-12 14:40:46.380545: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 67108864 totalling 256.00MiB
2024-07-12 14:40:46.380554: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 109510656 totalling 208.88MiB
2024-07-12 14:40:46.380563: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 5 Chunks of size 125960192 totalling 600.62MiB
2024-07-12 14:40:46.380572: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 128000000 totalling 122.07MiB
2024-07-12 14:40:46.380581: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 137552640 totalling 131.18MiB
2024-07-12 14:40:46.380591: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 238522880 totalling 227.47MiB
2024-07-12 14:40:46.380600: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 256000000 totalling 976.56MiB
2024-07-12 14:40:46.380609: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 411041792 totalling 1.53GiB
2024-07-12 14:40:46.380617: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 512000000 totalling 1.91GiB
2024-07-12 14:40:46.380626: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 3300000000 totalling 6.15GiB
2024-07-12 14:40:46.380635: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 13200000000 totalling 24.59GiB
2024-07-12 14:40:46.380644: I tensorflow/tsl/framework/bfc_allocator.cc:1107] Sum Total of in-use chunks: 37.35GiB
2024-07-12 14:40:46.380653: I tensorflow/tsl/framework/bfc_allocator.cc:1109] Total bytes in pool: 40231108608 memory_limit_: 40231108608 available bytes: 0 curr_region_allocation_bytes_: 80462217216
2024-07-12 14:40:46.380666: I tensorflow/tsl/framework/bfc_allocator.cc:1114] Stats: 
Limit:                     40231108608
InUse:                     40106093568
MaxInUse:                  40230617088
NumAllocs:                     2268042
MaxAllocSize:              13200000000
Reserved:                            0
PeakReserved:                        0
LargestFreeBlock:                    0

2024-07-12 14:40:46.380688: W tensorflow/tsl/framework/bfc_allocator.cc:497] ****************************************************************************************************
2024-07-12 14:40:46.380728: W tensorflow/core/framework/op_kernel.cc:1828] OP_REQUIRES failed at conv_ops_fused_impl.h:568 : RESOURCE_EXHAUSTED: OOM when allocating tensor with shape[32,256,62,62] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
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

Detected at node 'dual_encoder_all/model_1/conv2d_21/Relu_1' defined at (most recent call last):
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
Node: 'dual_encoder_all/model_1/conv2d_21/Relu_1'
OOM when allocating tensor with shape[32,256,62,62] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
	 [[{{node dual_encoder_all/model_1/conv2d_21/Relu_1}}]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info. This isn't available when running in Eager mode.
 [Op:__inference_train_function_1424895]
