[1;31m==>[0m Error: Spec '/xi3pch3' matches no installed packages.
Traceback (most recent call last):
  File "/home/xx4455/Comparable/Code/ImageExp/Experiments.py", line 104, in <module>
    experiment(dataName="FaceImage", col='Average', height=250, width=250)
  File "/home/xx4455/Comparable/Code/ImageExp/Experiments.py", line 22, in experiment
    protected_ts_AB_sex, protected_ts_AB_race_single, protected_ts_AB_sex_single) = dp.processData(
  File "/home/xx4455/Comparable/Code/ImageExp/DataProcessing.py", line 116, in processData
    data_tr['A'] = data_tr['A'].apply(retrievePixels).div(255.0)
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/pandas/core/series.py", line 4771, in apply
    return SeriesApply(self, func, convert_dtype, args, kwargs).apply()
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/pandas/core/apply.py", line 1123, in apply
    return self.apply_standard()
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/pandas/core/apply.py", line 1174, in apply_standard
    mapped = lib.map_infer(
  File "pandas/_libs/lib.pyx", line 2924, in pandas._libs.lib.map_infer
  File "/home/xx4455/Comparable/Code/ImageExp/DataProcessing.py", line 30, in retrievePixels
    img = tf.keras.utils.load_img(folder_path + path, target_size=(height, width))
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/keras/utils/image_utils.py", line 423, in load_img
    img = pil_image.open(io.BytesIO(f.read()))
  File "/.autofs/tools/spack/var/spack/environments/default-ml-24022101/.spack-env/view/lib/python3.9/site-packages/PIL/Image.py", line 3298, in open
    raise UnidentifiedImageError(msg)
PIL.UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x7f43a306d7c0>
