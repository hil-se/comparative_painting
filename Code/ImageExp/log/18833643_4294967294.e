[1;31m==>[0m Error: py-keras matches multiple packages.
  Matching packages:
    [0;90mjsj35ex[0m py-keras[0;36m@3.2.1[0m[0;32m%gcc[0m[0;32m@12.3.1[0m[0;35m arch=linux-rhel9-skylake_avx512[0m
    [0;90mukcde6j[0m py-keras[0;36m@3.2.1[0m[0;32m%gcc[0m[0;32m@12.3.1[0m[0;35m arch=linux-rhel9-skylake_avx512[0m
    [0;90m7b6blx5[0m py-keras[0;36m@3.2.1[0m[0;32m%gcc[0m[0;32m@12.3.1[0m[0;35m arch=linux-rhel9-skylake_avx512[0m
  Use a more specific spec (e.g., prepend '/' to the hash).
2024-07-12 23:27:17.193268: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38367 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:af:00.0, compute capability: 8.0
2024-07-12 23:27:25.822597: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel/dropout/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-12 23:27:26.632568: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:432] Loaded cuDNN version 8700
2024-07-12 23:27:28.993296: I tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:606] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2024-07-12 23:27:32.451471: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f410526c9a0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2024-07-12 23:27:32.451520: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB, Compute Capability 8.0
2024-07-12 23:27:32.531081: I ./tensorflow/compiler/jit/device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
2024-07-13 00:07:08.725694: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape indual_encoder_all/model_1/dropout_2/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-13 01:15:32.149951: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel_2/dropout_4/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
2024-07-13 01:58:23.985593: W tensorflow/tsl/framework/bfc_allocator.cc:485] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.07GiB (rounded to 3300000000)requested by op _EagerConst
If the cause is memory fragmentation maybe the environment variable 'TF_GPU_ALLOCATOR=cuda_malloc_async' will improve the situation. 
Current allocation summary follows.
Current allocation summary follows.
2024-07-13 01:58:24.146310: I tensorflow/tsl/framework/bfc_allocator.cc:1039] BFCAllocator dump for GPU_0_bfc
2024-07-13 01:58:24.146351: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (256): 	Total Chunks: 68, Chunks in use: 66. 17.0KiB allocated for chunks. 16.5KiB in use in bin. 1.4KiB client-requested in use in bin.
2024-07-13 01:58:24.146365: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (512): 	Total Chunks: 4, Chunks in use: 4. 2.0KiB allocated for chunks. 2.0KiB in use in bin. 2.0KiB client-requested in use in bin.
2024-07-13 01:58:24.146375: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (1024): 	Total Chunks: 12, Chunks in use: 11. 12.8KiB allocated for chunks. 11.5KiB in use in bin. 11.0KiB client-requested in use in bin.
2024-07-13 01:58:24.146385: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (2048): 	Total Chunks: 14, Chunks in use: 12. 33.8KiB allocated for chunks. 28.0KiB in use in bin. 24.0KiB client-requested in use in bin.
2024-07-13 01:58:24.146394: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (4096): 	Total Chunks: 1, Chunks in use: 1. 6.8KiB allocated for chunks. 6.8KiB in use in bin. 6.8KiB client-requested in use in bin.
2024-07-13 01:58:24.146404: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (8192): 	Total Chunks: 3, Chunks in use: 2. 34.8KiB allocated for chunks. 21.0KiB in use in bin. 17.0KiB client-requested in use in bin.
2024-07-13 01:58:24.146428: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (16384): 	Total Chunks: 4, Chunks in use: 4. 75.8KiB allocated for chunks. 75.8KiB in use in bin. 64.0KiB client-requested in use in bin.
2024-07-13 01:58:24.146438: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (32768): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2024-07-13 01:58:24.146450: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (65536): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2024-07-13 01:58:24.146460: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (131072): 	Total Chunks: 5, Chunks in use: 3. 863.2KiB allocated for chunks. 491.0KiB in use in bin. 425.5KiB client-requested in use in bin.
2024-07-13 01:58:24.146470: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (262144): 	Total Chunks: 2, Chunks in use: 2. 576.0KiB allocated for chunks. 576.0KiB in use in bin. 576.0KiB client-requested in use in bin.
2024-07-13 01:58:24.146479: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (524288): 	Total Chunks: 70, Chunks in use: 64. 50.34MiB allocated for chunks. 45.60MiB in use in bin. 45.47MiB client-requested in use in bin.
2024-07-13 01:58:24.146489: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (1048576): 	Total Chunks: 4, Chunks in use: 4. 4.79MiB allocated for chunks. 4.79MiB in use in bin. 3.68MiB client-requested in use in bin.
2024-07-13 01:58:24.146498: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (2097152): 	Total Chunks: 3, Chunks in use: 3. 8.50MiB allocated for chunks. 8.50MiB in use in bin. 6.75MiB client-requested in use in bin.
2024-07-13 01:58:24.146507: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (4194304): 	Total Chunks: 5, Chunks in use: 5. 21.00MiB allocated for chunks. 21.00MiB in use in bin. 19.25MiB client-requested in use in bin.
2024-07-13 01:58:24.146517: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (8388608): 	Total Chunks: 10, Chunks in use: 10. 96.75MiB allocated for chunks. 96.75MiB in use in bin. 90.00MiB client-requested in use in bin.
2024-07-13 01:58:24.146526: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (16777216): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2024-07-13 01:58:24.146535: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (33554432): 	Total Chunks: 2, Chunks in use: 1. 100.14MiB allocated for chunks. 60.71MiB in use in bin. 40.97MiB client-requested in use in bin.
2024-07-13 01:58:24.146545: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (67108864): 	Total Chunks: 3, Chunks in use: 2. 281.75MiB allocated for chunks. 179.69MiB in use in bin. 128.00MiB client-requested in use in bin.
2024-07-13 01:58:24.146554: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (134217728): 	Total Chunks: 1, Chunks in use: 0. 165.75MiB allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2024-07-13 01:58:24.146564: I tensorflow/tsl/framework/bfc_allocator.cc:1046] Bin (268435456): 	Total Chunks: 9, Chunks in use: 6. 36.75GiB allocated for chunks. 29.19GiB in use in bin. 29.19GiB client-requested in use in bin.
2024-07-13 01:58:24.146580: I tensorflow/tsl/framework/bfc_allocator.cc:1062] Bin for 3.07GiB was 256.00MiB, Chunk State: 
2024-07-13 01:58:24.146604: I tensorflow/tsl/framework/bfc_allocator.cc:1068]   Size: 1.75GiB | Requested Size: 392.00MiB | in_use: 0 | bin_num: 20, prev:   Size: 786.78MiB | Requested Size: 786.78MiB | in_use: 1 | bin_num: -1, next:   Size: 256B | Requested Size: 8B | in_use: 1 | bin_num: -1
2024-07-13 01:58:24.146630: I tensorflow/tsl/framework/bfc_allocator.cc:1068]   Size: 2.78GiB | Requested Size: 128B | in_use: 0 | bin_num: 20, prev:   Size: 3.07GiB | Requested Size: 3.07GiB | in_use: 1 | bin_num: -1, next:   Size: 256B | Requested Size: 256B | in_use: 1 | bin_num: -1
2024-07-13 01:58:24.146650: I tensorflow/tsl/framework/bfc_allocator.cc:1068]   Size: 3.03GiB | Requested Size: 488.28MiB | in_use: 0 | bin_num: 20, prev:   Size: 392.00MiB | Requested Size: 392.00MiB | in_use: 1 | bin_num: -1, next:   Size: 6.8KiB | Requested Size: 6.8KiB | in_use: 1 | bin_num: -1
2024-07-13 01:58:24.146659: I tensorflow/tsl/framework/bfc_allocator.cc:1075] Next region of size 40231108608
2024-07-13 01:58:24.146670: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2000000 of size 256 next 1
2024-07-13 01:58:24.146681: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2000100 of size 1280 next 2
2024-07-13 01:58:24.146689: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2000600 of size 256 next 3
2024-07-13 01:58:24.146697: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2000700 of size 256 next 4
2024-07-13 01:58:24.146705: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2000800 of size 256 next 6
2024-07-13 01:58:24.146713: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2000900 of size 256 next 7
2024-07-13 01:58:24.146720: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2000a00 of size 256 next 5
2024-07-13 01:58:24.146728: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2000b00 of size 256 next 8
2024-07-13 01:58:24.146736: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2000c00 of size 256 next 13
2024-07-13 01:58:24.146744: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2000d00 of size 256 next 11
2024-07-13 01:58:24.146752: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2000e00 of size 256 next 12
2024-07-13 01:58:24.146760: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2000f00 of size 512 next 16
2024-07-13 01:58:24.146773: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2001100 of size 256 next 17
2024-07-13 01:58:24.146781: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2001200 of size 256 next 20
2024-07-13 01:58:24.146789: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2001300 of size 256 next 58
2024-07-13 01:58:24.146796: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2001400 of size 256 next 23
2024-07-13 01:58:24.146804: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2001500 of size 256 next 21
2024-07-13 01:58:24.146812: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2001600 of size 256 next 22
2024-07-13 01:58:24.146820: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2001700 of size 1024 next 26
2024-07-13 01:58:24.146830: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2001b00 of size 256 next 27
2024-07-13 01:58:24.146838: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2001c00 of size 256 next 30
2024-07-13 01:58:24.146846: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2001d00 of size 1024 next 154
2024-07-13 01:58:24.146854: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2002100 of size 1024 next 36
2024-07-13 01:58:24.146862: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2002500 of size 256 next 31
2024-07-13 01:58:24.146870: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2002600 of size 256 next 32
2024-07-13 01:58:24.146878: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2002700 of size 2048 next 38
2024-07-13 01:58:24.146889: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2002f00 of size 256 next 39
2024-07-13 01:58:24.146897: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2003000 of size 256 next 42
2024-07-13 01:58:24.146904: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2003100 of size 2048 next 193
2024-07-13 01:58:24.146912: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2003900 of size 2048 next 173
2024-07-13 01:58:24.146920: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2004100 of size 2048 next 214
2024-07-13 01:58:24.146932: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2004900 of size 2048 next 120
2024-07-13 01:58:24.146940: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2005100 of size 3840 next 52
2024-07-13 01:58:24.146952: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2006000 of size 256 next 54
2024-07-13 01:58:24.146960: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2006100 of size 256 next 72
2024-07-13 01:58:24.146968: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2006200 of size 256 next 77
2024-07-13 01:58:24.146975: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2006300 of size 256 next 78
2024-07-13 01:58:24.146983: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2006400 of size 256 next 79
2024-07-13 01:58:24.146991: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2006500 of size 3072 next 70
2024-07-13 01:58:24.146999: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2007100 of size 256 next 71
2024-07-13 01:58:24.147007: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2007200 of size 256 next 167
2024-07-13 01:58:24.147015: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2007300 of size 256 next 53
2024-07-13 01:58:24.147023: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2007400 of size 256 next 48
2024-07-13 01:58:24.147031: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2007500 of size 256 next 43
2024-07-13 01:58:24.147039: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2007600 of size 2048 next 132
2024-07-13 01:58:24.147047: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2007e00 of size 2048 next 145
2024-07-13 01:58:24.147055: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2008600 of size 256 next 170
2024-07-13 01:58:24.147062: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2008700 of size 256 next 171
2024-07-13 01:58:24.147070: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2008800 of size 512 next 184
2024-07-13 01:58:24.147078: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2008a00 of size 512 next 174
2024-07-13 01:58:24.147086: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2008c00 of size 1280 next 334
2024-07-13 01:58:24.147094: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2009100 of size 512 next 160
2024-07-13 01:58:24.147102: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2009300 of size 1024 next 144
2024-07-13 01:58:24.147110: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2009700 of size 2048 next 156
2024-07-13 01:58:24.147118: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b2009f00 of size 1024 next 168
2024-07-13 01:58:24.147126: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b200a300 of size 1024 next 187
2024-07-13 01:58:24.147133: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b200a700 of size 1024 next 222
2024-07-13 01:58:24.147141: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b200ab00 of size 2816 next 57
2024-07-13 01:58:24.147150: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b200b600 of size 256 next 55
2024-07-13 01:58:24.147158: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b200b700 of size 256 next 56
2024-07-13 01:58:24.147166: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b200b800 of size 16384 next 62
2024-07-13 01:58:24.147174: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b200f800 of size 256 next 60
2024-07-13 01:58:24.147182: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b200f900 of size 256 next 61
2024-07-13 01:58:24.147190: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b200fa00 of size 67108864 next 166
2024-07-13 01:58:24.147199: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b600fa00 of size 1179648 next 201
2024-07-13 01:58:24.147211: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b612fa00 of size 3014656 next 150
2024-07-13 01:58:24.147221: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b640fa00 of size 4194304 next 139
2024-07-13 01:58:24.147232: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b680fa00 of size 4718592 next 205
2024-07-13 01:58:24.147240: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b6c8fa00 of size 9437184 next 213
2024-07-13 01:58:24.147250: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b758fa00 of size 9437184 next 153
2024-07-13 01:58:24.147258: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b7e8fa00 of size 9437184 next 217
2024-07-13 01:58:24.147266: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b878fa00 of size 9437184 next 221
2024-07-13 01:58:24.147274: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b908fa00 of size 9437184 next 133
2024-07-13 01:58:24.147282: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b998fa00 of size 4194304 next 136
2024-07-13 01:58:24.147290: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b9d8fa00 of size 750080 next 93
2024-07-13 01:58:24.147298: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b9e46c00 of size 750080 next 10
2024-07-13 01:58:24.147306: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b9efde00 of size 750080 next 86
2024-07-13 01:58:24.147314: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41b9fb5000 of size 750080 next 231
2024-07-13 01:58:24.147322: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41ba06c200 of size 750080 next 101
2024-07-13 01:58:24.147330: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41ba123400 of size 750080 next 209
2024-07-13 01:58:24.147338: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41ba1da600 of size 750080 next 104
2024-07-13 01:58:24.147346: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41ba291800 of size 750080 next 94
2024-07-13 01:58:24.147353: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41ba348a00 of size 750080 next 134
2024-07-13 01:58:24.147361: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41ba3ffc00 of size 750080 next 185
2024-07-13 01:58:24.147369: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f41ba4b6e00 of size 750080 next 89
2024-07-13 01:58:24.147377: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41ba56e000 of size 750080 next 161
2024-07-13 01:58:24.147385: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41ba625200 of size 750080 next 82
2024-07-13 01:58:24.147393: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41ba6dc400 of size 750080 next 88
2024-07-13 01:58:24.147401: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41ba793600 of size 750080 next 311
2024-07-13 01:58:24.147409: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41ba84a800 of size 750080 next 110
2024-07-13 01:58:24.147416: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41ba901a00 of size 750080 next 322
2024-07-13 01:58:24.147424: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41ba9b8c00 of size 750080 next 225
2024-07-13 01:58:24.147432: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41baa6fe00 of size 750080 next 98
2024-07-13 01:58:24.147440: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bab27000 of size 750080 next 91
2024-07-13 01:58:24.147448: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41babde200 of size 750080 next 226
2024-07-13 01:58:24.147456: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bac95400 of size 750080 next 124
2024-07-13 01:58:24.147463: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bad4c600 of size 750080 next 212
2024-07-13 01:58:24.147471: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bae03800 of size 750080 next 232
2024-07-13 01:58:24.147483: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41baebaa00 of size 750080 next 177
2024-07-13 01:58:24.147491: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41baf71c00 of size 750080 next 68
2024-07-13 01:58:24.147499: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bb028e00 of size 750080 next 51
2024-07-13 01:58:24.147507: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bb0e0000 of size 750080 next 290
2024-07-13 01:58:24.147514: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bb197200 of size 750080 next 204
2024-07-13 01:58:24.147522: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bb24e400 of size 750080 next 274
2024-07-13 01:58:24.147530: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bb305600 of size 750080 next 224
2024-07-13 01:58:24.147538: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bb3bc800 of size 750080 next 268
2024-07-13 01:58:24.147546: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bb473a00 of size 750080 next 220
2024-07-13 01:58:24.147554: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bb52ac00 of size 750080 next 197
2024-07-13 01:58:24.147561: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bb5e1e00 of size 750080 next 228
2024-07-13 01:58:24.147569: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bb699000 of size 750080 next 200
2024-07-13 01:58:24.147577: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bb750200 of size 750080 next 291
2024-07-13 01:58:24.147585: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bb807400 of size 750080 next 196
2024-07-13 01:58:24.147593: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bb8be600 of size 750080 next 45
2024-07-13 01:58:24.147601: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bb975800 of size 750080 next 105
2024-07-13 01:58:24.147609: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bba2ca00 of size 750080 next 218
2024-07-13 01:58:24.147628: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bbae3c00 of size 750080 next 179
2024-07-13 01:58:24.147636: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bbb9ae00 of size 750080 next 131
2024-07-13 01:58:24.147644: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bbc52000 of size 750080 next 130
2024-07-13 01:58:24.147652: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bbd09200 of size 750080 next 255
2024-07-13 01:58:24.147660: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bbdc0400 of size 750080 next 103
2024-07-13 01:58:24.147668: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bbe77600 of size 750080 next 99
2024-07-13 01:58:24.147676: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bbf2e800 of size 750080 next 207
2024-07-13 01:58:24.147684: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bbfe5a00 of size 750080 next 37
2024-07-13 01:58:24.147692: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc09cc00 of size 750080 next 188
2024-07-13 01:58:24.147700: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc153e00 of size 750080 next 69
2024-07-13 01:58:24.147708: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc20b000 of size 750080 next 181
2024-07-13 01:58:24.147715: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc2c2200 of size 750080 next 158
2024-07-13 01:58:24.147723: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc379400 of size 750080 next 202
2024-07-13 01:58:24.147731: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc430600 of size 750080 next 123
2024-07-13 01:58:24.147739: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc4e7800 of size 750080 next 96
2024-07-13 01:58:24.147747: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc59ea00 of size 750080 next 318
2024-07-13 01:58:24.147758: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc655c00 of size 750080 next 251
2024-07-13 01:58:24.147766: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f41bc70ce00 of size 882944 next 215
2024-07-13 01:58:24.147774: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc7e4700 of size 256 next 198
2024-07-13 01:58:24.147782: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc7e4800 of size 256 next 203
2024-07-13 01:58:24.147790: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f41bc7e4900 of size 256 next 229
2024-07-13 01:58:24.147798: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc7e4a00 of size 256 next 230
2024-07-13 01:58:24.147806: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f41bc7e4b00 of size 1280 next 192
2024-07-13 01:58:24.147814: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc7e5000 of size 147456 next 194
2024-07-13 01:58:24.147824: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc809000 of size 16384 next 151
2024-07-13 01:58:24.147832: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc80d000 of size 16384 next 122
2024-07-13 01:58:24.147840: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc811000 of size 1024 next 129
2024-07-13 01:58:24.147848: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc811400 of size 1024 next 64
2024-07-13 01:58:24.147856: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc811800 of size 256 next 63
2024-07-13 01:58:24.147864: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc811900 of size 256 next 9
2024-07-13 01:58:24.147872: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc811a00 of size 256 next 66
2024-07-13 01:58:24.147880: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc811b00 of size 256 next 67
2024-07-13 01:58:24.147888: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc811c00 of size 256 next 65
2024-07-13 01:58:24.147896: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc811d00 of size 256 next 15
2024-07-13 01:58:24.147903: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc811e00 of size 256 next 18
2024-07-13 01:58:24.147911: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc811f00 of size 256 next 19
2024-07-13 01:58:24.147919: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc812000 of size 256 next 24
2024-07-13 01:58:24.147927: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc812100 of size 256 next 14
2024-07-13 01:58:24.147935: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc812200 of size 256 next 33
2024-07-13 01:58:24.147943: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc812300 of size 256 next 49
2024-07-13 01:58:24.147951: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f41bc812400 of size 994304 next 121
2024-07-13 01:58:24.147959: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc905000 of size 294912 next 189
2024-07-13 01:58:24.147967: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc94d000 of size 294912 next 172
2024-07-13 01:58:24.147975: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bc995000 of size 589824 next 152
2024-07-13 01:58:24.147984: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bca25000 of size 721920 next 138
2024-07-13 01:58:24.147992: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bcad5400 of size 256 next 162
2024-07-13 01:58:24.147999: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f41bcad5500 of size 2048 next 210
2024-07-13 01:58:24.148007: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bcad5d00 of size 2560 next 148
2024-07-13 01:58:24.148018: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bcad6700 of size 256 next 142
2024-07-13 01:58:24.148029: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f41bcad6800 of size 14080 next 169
2024-07-13 01:58:24.148037: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bcad9f00 of size 214528 next 125
2024-07-13 01:58:24.148045: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f41bcb0e500 of size 3840 next 176
2024-07-13 01:58:24.148053: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bcb0f400 of size 10496 next 165
2024-07-13 01:58:24.148061: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bcb11d00 of size 11008 next 219
2024-07-13 01:58:24.148069: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f41bcb14800 of size 840704 next 25
2024-07-13 01:58:24.148077: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f41bcbe1c00 of size 411041792 next 159
2024-07-13 01:58:24.148086: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f41d53e1c00 of size 3255434240 next 206
2024-07-13 01:58:24.148094: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4297481800 of size 6912 next 146
2024-07-13 01:58:24.148102: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4297483300 of size 28416 next 126
2024-07-13 01:58:24.148110: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f429748a200 of size 1464832 next 180
2024-07-13 01:58:24.148118: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f42975efc00 of size 1179648 next 208
2024-07-13 01:58:24.148126: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f429770fc00 of size 3538944 next 135
2024-07-13 01:58:24.148134: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4297a6fc00 of size 16515072 next 163
2024-07-13 01:58:24.148142: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4298a2fc00 of size 4718592 next 140
2024-07-13 01:58:24.148150: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4298eafc00 of size 2359296 next 183
2024-07-13 01:58:24.148160: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f42990efc00 of size 256 next 74
2024-07-13 01:58:24.148168: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f42990efd00 of size 256 next 84
2024-07-13 01:58:24.148176: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f42990efe00 of size 139008 next 90
2024-07-13 01:58:24.148184: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4299111d00 of size 256 next 143
2024-07-13 01:58:24.148192: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4299111e00 of size 256 next 83
2024-07-13 01:58:24.148200: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4299111f00 of size 256 next 85
2024-07-13 01:58:24.148208: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4299112000 of size 256 next 28
2024-07-13 01:58:24.148216: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4299112100 of size 256 next 44
2024-07-13 01:58:24.148223: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f4299112200 of size 256 next 35
2024-07-13 01:58:24.148231: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4299112300 of size 256 next 87
2024-07-13 01:58:24.148239: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4299112400 of size 140800 next 248
2024-07-13 01:58:24.148247: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f4299134a00 of size 242176 next 73
2024-07-13 01:58:24.148255: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f429916fc00 of size 256 next 29
2024-07-13 01:58:24.148263: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f429916fd00 of size 825000192 next 237
2024-07-13 01:58:24.148274: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f42ca437e00 of size 1873822976 next 80
2024-07-13 01:58:24.148281: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4339f3c300 of size 256 next 81
2024-07-13 01:58:24.148290: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4339f3c400 of size 13200000000 next 247
2024-07-13 01:58:24.148298: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f464cbbc800 of size 13200000000 next 250
2024-07-13 01:58:24.148309: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f495f83cc00 of size 3300000000 next 97
2024-07-13 01:58:24.148320: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f4a2435cd00 of size 2988683008 next 321
2024-07-13 01:58:24.148327: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4ad6597c00 of size 256 next 284
2024-07-13 01:58:24.148335: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4ad6597d00 of size 750080 next 233
2024-07-13 01:58:24.148343: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4ad664ef00 of size 750080 next 34
2024-07-13 01:58:24.148351: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4ad6706100 of size 750080 next 100
2024-07-13 01:58:24.148359: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f4ad67bd300 of size 41344768 next 178
2024-07-13 01:58:24.148367: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4ad8f2b200 of size 121307136 next 227
2024-07-13 01:58:24.148375: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4ae02db200 of size 750080 next 304
2024-07-13 01:58:24.148383: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f4ae0392400 of size 750080 next 199
2024-07-13 01:58:24.148391: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4ae0449600 of size 750080 next 211
2024-07-13 01:58:24.148399: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f4ae0500800 of size 750080 next 336
2024-07-13 01:58:24.148407: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4ae05b7a00 of size 1193984 next 182
2024-07-13 01:58:24.148415: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4ae06db200 of size 4194304 next 195
2024-07-13 01:58:24.148423: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f4ae0adb200 of size 107020288 next 223
2024-07-13 01:58:24.148431: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4ae70eb200 of size 411041792 next 186
2024-07-13 01:58:24.148439: I tensorflow/tsl/framework/bfc_allocator.cc:1095] Free  at 7f4aff8eb200 of size 173801472 next 270
2024-07-13 01:58:24.148447: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4b09eab200 of size 9437184 next 128
2024-07-13 01:58:24.148455: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4b0a7ab200 of size 9437184 next 127
2024-07-13 01:58:24.148463: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4b0b0ab200 of size 9437184 next 164
2024-07-13 01:58:24.148471: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4b0b9ab200 of size 9437184 next 157
2024-07-13 01:58:24.148484: I tensorflow/tsl/framework/bfc_allocator.cc:1095] InUse at 7f4b0c2ab200 of size 63655424 next 18446744073709551615
2024-07-13 01:58:24.148492: I tensorflow/tsl/framework/bfc_allocator.cc:1100]      Summary of in-use Chunks by size: 
2024-07-13 01:58:24.148502: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 66 Chunks of size 256 totalling 16.5KiB
2024-07-13 01:58:24.148511: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 4 Chunks of size 512 totalling 2.0KiB
2024-07-13 01:58:24.148519: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 9 Chunks of size 1024 totalling 9.0KiB
2024-07-13 01:58:24.148528: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 1280 totalling 2.5KiB
2024-07-13 01:58:24.148536: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 8 Chunks of size 2048 totalling 16.0KiB
2024-07-13 01:58:24.148545: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 2560 totalling 2.5KiB
2024-07-13 01:58:24.148553: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 2816 totalling 2.8KiB
2024-07-13 01:58:24.148561: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 3072 totalling 3.0KiB
2024-07-13 01:58:24.148569: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 3840 totalling 3.8KiB
2024-07-13 01:58:24.148578: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 6912 totalling 6.8KiB
2024-07-13 01:58:24.148589: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 10496 totalling 10.2KiB
2024-07-13 01:58:24.148598: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 11008 totalling 10.8KiB
2024-07-13 01:58:24.148607: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 3 Chunks of size 16384 totalling 48.0KiB
2024-07-13 01:58:24.148627: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 28416 totalling 27.8KiB
2024-07-13 01:58:24.148637: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 140800 totalling 137.5KiB
2024-07-13 01:58:24.148645: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 147456 totalling 144.0KiB
2024-07-13 01:58:24.148654: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 214528 totalling 209.5KiB
2024-07-13 01:58:24.148662: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 294912 totalling 576.0KiB
2024-07-13 01:58:24.148671: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 589824 totalling 576.0KiB
2024-07-13 01:58:24.148679: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 721920 totalling 705.0KiB
2024-07-13 01:58:24.148688: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 62 Chunks of size 750080 totalling 44.35MiB
2024-07-13 01:58:24.148696: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 1179648 totalling 2.25MiB
2024-07-13 01:58:24.148705: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 1193984 totalling 1.14MiB
2024-07-13 01:58:24.148713: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 1464832 totalling 1.40MiB
2024-07-13 01:58:24.148721: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 2359296 totalling 2.25MiB
2024-07-13 01:58:24.148729: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 3014656 totalling 2.88MiB
2024-07-13 01:58:24.148738: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 3538944 totalling 3.38MiB
2024-07-13 01:58:24.148746: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 3 Chunks of size 4194304 totalling 12.00MiB
2024-07-13 01:58:24.148755: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 4718592 totalling 9.00MiB
2024-07-13 01:58:24.148763: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 9 Chunks of size 9437184 totalling 81.00MiB
2024-07-13 01:58:24.148772: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 16515072 totalling 15.75MiB
2024-07-13 01:58:24.148780: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 63655424 totalling 60.71MiB
2024-07-13 01:58:24.148789: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 67108864 totalling 64.00MiB
2024-07-13 01:58:24.148797: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 121307136 totalling 115.69MiB
2024-07-13 01:58:24.148806: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 411041792 totalling 784.00MiB
2024-07-13 01:58:24.148814: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 825000192 totalling 786.78MiB
2024-07-13 01:58:24.148823: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 1 Chunks of size 3300000000 totalling 3.07GiB
2024-07-13 01:58:24.148831: I tensorflow/tsl/framework/bfc_allocator.cc:1103] 2 Chunks of size 13200000000 totalling 24.59GiB
2024-07-13 01:58:24.148840: I tensorflow/tsl/framework/bfc_allocator.cc:1107] Sum Total of in-use chunks: 29.60GiB
2024-07-13 01:58:24.148848: I tensorflow/tsl/framework/bfc_allocator.cc:1109] Total bytes in pool: 40231108608 memory_limit_: 40231108608 available bytes: 0 curr_region_allocation_bytes_: 80462217216
2024-07-13 01:58:24.148860: I tensorflow/tsl/framework/bfc_allocator.cc:1114] Stats: 
Limit:                     40231108608
InUse:                     31785630720
MaxInUse:                  37595815424
NumAllocs:                    21974606
MaxAllocSize:              13200000000
Reserved:                            0
PeakReserved:                        0
LargestFreeBlock:                    0

2024-07-13 01:58:24.148883: W tensorflow/tsl/framework/bfc_allocator.cc:497] **_______***____***************************************************************************______***
Traceback (most recent call last):
  File "/home/xx4455/Comparable/Code/ImageExp/Experiments.py", line 104, in <module>
    experiment(dataName="FaceImage", col='Average', height=250, width=250)
  File "/home/xx4455/Comparable/Code/ImageExp/Experiments.py", line 63, in experiment
    MI_encoder_sex_single) = cl.comparabilityExperiment(
  File "/home/xx4455/Comparable/Code/ImageExp/Classification.py", line 228, in comparabilityExperiment
    dual_encoder = train_model(train=train, val=val, y_true=y_true, shared=True, height=height, width=width)
  File "/home/xx4455/Comparable/Code/ImageExp/Classification.py", line 66, in train_model
    dual_encoder = learn(train, epochs=epochs, validation_data=val, y_true=y_true, shared=shared, height=height,
  File "/home/xx4455/Comparable/Code/ImageExp/Classification.py", line 38, in learn
    val_dataset = tf.data.Dataset.from_tensor_slices(v_feature)
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/tensorflow/python/data/ops/dataset_ops.py", line 831, in from_tensor_slices
    return from_tensor_slices_op._from_tensor_slices(tensors, name)
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/tensorflow/python/data/ops/from_tensor_slices_op.py", line 25, in _from_tensor_slices
    return _TensorSliceDataset(tensors, name=name)
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/tensorflow/python/data/ops/from_tensor_slices_op.py", line 33, in __init__
    element = structure.normalize_element(element)
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/tensorflow/python/data/util/structure.py", line 133, in normalize_element
    ops.convert_to_tensor(t, name="component_%d" % i, dtype=dtype))
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/tensorflow/python/profiler/trace.py", line 183, in wrapped
    return func(*args, **kwargs)
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/tensorflow/python/framework/ops.py", line 1443, in convert_to_tensor
    return tensor_conversion_registry.convert(
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/tensorflow/python/framework/tensor_conversion_registry.py", line 234, in convert
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/tensorflow/python/framework/constant_op.py", line 324, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/tensorflow/python/framework/constant_op.py", line 263, in constant
    return _constant_impl(value, dtype, shape, name, verify_shape=False,
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/tensorflow/python/framework/constant_op.py", line 275, in _constant_impl
    return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/tensorflow/python/framework/constant_op.py", line 285, in _constant_eager_impl
    t = convert_to_eager_tensor(value, ctx, dtype)
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/tensorflow/python/framework/constant_op.py", line 98, in convert_to_eager_tensor
    return ops.EagerTensor(value, ctx.device_name, dtype)
tensorflow.python.framework.errors_impl.InternalError: Failed copying input tensor from /job:localhost/replica:0/task:0/device:CPU:0 to /job:localhost/replica:0/task:0/device:GPU:0 in order to run _EagerConst: Dst tensor is not initialized.
