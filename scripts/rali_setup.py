from enum import Enum
from amd.rali.plugin.pytorch import RALIClassificationIterator
from amd.rali.plugin.pytorch import RALI_iterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types
import numpy as np
from ctypes import create_string_buffer

#batch size = 64
raliList_mode1_64 = ['original', 'warpAffine', 'contrast', 'rain', 
            'brightness', 'colorTemp', 'exposure', 'vignette', 
            'blur', 'snow', 'pixelate', 'SnPNoise', 
            'gamma', 'rotate', 'flip', 'blend',
            'rotate45+resize', 'rotate45+warpAffine', 'rotate45+contrast', 'rotate45+rain', 
            'rotate45+brightness', 'rotate45+colorTemp', 'rotate45+exposure', 'rotate45+vignette', 
            'rotate45+blur', 'rotate45+snow', 'rotate45+pixelate', 'rotate45+SnPNoise', 
            'rotate45+gamma', 'rotate45+rotate', 'rotate45+flip', 'rotate45+blend',
            'flip+resize', 'flip+warpAffine', 'flip+contrast', 'flip+rain', 
            'flip+brightness', 'flip+colorTemp', 'flip+exposure', 'flip+vignette', 
            'flip+blur', 'flip+snow', 'flip+pixelate', 'flip+SnPNoise', 
            'flip+gamma', 'flip+rotate', 'flip+flip', 'flip+blend',         
            'rotate150+resize', 'rotate150+warpAffine', 'rotate150+contrast', 'rotate150+rain', 
            'rotate150+brightness', 'rotate150+colorTemp', 'rotate150+exposure', 'rotate150+vignette', 
            'rotate150+blur', 'rotate150+snow', 'rotate150+pixelate', 'rotate150+SnPNoise', 
            'rotate150+gamma', 'rotate150+rotate', 'rotate150+flip', 'rotate150+blend']
raliList_mode2_64 = ['original', 'warpAffine', 'contrast', 'rain', 
            'brightness', 'colorTemp', 'exposure', 'vignette', 
            'blur', 'snow', 'pixelate', 'SnPNoise', 
            'gamma', 'rotate', 'flip', 'blend',
            'warpAffine+original', 'warpAffine+warpAffine', 'warpAffine+contrast', 'warpAffine+rain', 
            'warpAffine+brightness', 'warpAffine+colorTemp', 'warpAffine+exposure', 'warpAffine+vignette', 
            'warpAffine+blur', 'warpAffine+snow', 'pixelate', 'warpAffine+SnPNoise', 
            'warpAffine+gamma', 'warpAffine+rotate', 'warpAffine+flip', 'warpAffine+blend',
            'fishEye+original', 'fishEye+warpAffine', 'fishEye+contrast', 'fishEye+rain', 
            'fishEye+brightness', 'fishEye+colorTemp', 'fishEye+exposure', 'fishEye+vignette', 
            'fishEye+blur', 'fishEye+snow', 'fishEye+pixelate', 'fishEye+SnPNoise', 
            'fishEye+gamma', 'fishEye+rotate', 'fishEye+flip', 'fishEye+blend',
            'lensCorrection+original', 'lensCorrection+warpAffine', 'lensCorrection+contrast', 'lensCorrection+rain', 
            'lensCorrection+brightness', 'lensCorrection+colorTemp', 'exposure', 'lensCorrection+vignette', 
            'lensCorrection+blur', 'lensCorrection+snow', 'lensCorrection+pixelate', 'lensCorrection+SnPNoise', 
            'lensCorrection+gamma', 'lensCorrection+rotate', 'lensCorrection+flip', 'lensCorrection+blend',]
raliList_mode3_64 = ['original', 'warpAffine', 'contrast', 'rain', 
            'brightness', 'colorTemp', 'exposure', 'vignette', 
            'blur', 'snow', 'pixelate', 'SnPNoise', 
            'gamma', 'rotate', 'flip', 'blend',
            'colorTemp+original', 'colorTemp+warpAffine', 'colorTemp+contrast', 'colorTemp+rain', 
            'colorTemp+brightness', 'colorTemp+colorTemp', 'colorTemp+exposure', 'colorTemp+vignette', 
            'colorTemp+blur', 'colorTemp+snow', 'colorTemp+pixelate', 'colorTemp+SnPNoise', 
            'colorTemp+gamma', 'colorTemp+rotate', 'colorTemp+flip', 'colorTemp+blend',
            'colorTemp+original', 'colorTemp+warpAffine', 'colorTemp+contrast', 'colorTemp+rain', 
            'colorTemp+brightness', 'colorTemp+colorTemp', 'colorTemp+exposure', 'colorTemp+vignette', 
            'colorTemp+blur', 'colorTemp+snow', 'colorTemp+pixelate', 'colorTemp+SnPNoise', 
            'colorTemp+gamma', 'colorTemp+rotate', 'colorTemp+flip', 'colorTemp+blend',
            'warpAffine+original', 'warpAffine+warpAffine', 'warpAffine+contrast', 'warpAffine+rain', 
            'warpAffine+brightness', 'warpAffine+colorTemp', 'warpAffine+exposure', 'warpAffine+vignette', 
            'warpAffine+blur', 'warpAffine+snow', 'pixelate', 'warpAffine+SnPNoise', 
            'warpAffine+gamma', 'warpAffine+rotate', 'warpAffine+flip', 'warpAffine+blend']
raliList_mode4_64 = ['original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original']
raliList_mode5_64 = ['original', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop']
#batch size = 16
raliList_mode1_16 = ['original', 'warpAffine', 'contrast', 'rain', 
            'brightness', 'colorTemp', 'exposure', 'vignette', 
            'blur', 'snow', 'pixelate', 'SnPNoise', 
            'gamma', 'rotate', 'flip', 'blend']
raliList_mode2_16 = ['original', 'warpAffine', 'contrast', 'contrast+rain', 
            'brightness', 'brightness+colorTemp', 'exposure', 'exposure+vignette', 
            'blur', 'blur+snow', 'pixelate', 'pixelate+SnPNoise', 
            'gamma', 'rotate', 'rotate+flip', 'blend']
raliList_mode3_16 = ['original', 'warpAffine', 'contrast', 'warpAffine+rain', 
            'brightness', 'colorTemp', 'exposure', 'vignette', 
            'blur', 'vignette+snow', 'pixelate', 'gamma',
            'SnPNoise+gamma', 'rotate', 'flip+pixelate', 'blend']
raliList_mode4_16 = ['original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original',
                    'original', 'original', 'original', 'original']
raliList_mode5_16 = ['original', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop',
                    'nop', 'nop', 'nop', 'nop']

# Class to initialize Rali and call the augmentations 
class InferencePipe(Pipeline):
    def __init__(self, image_validation, model_batch_size, raliMode, c_img, h_img, w_img, rali_batch_size, tensor_dtype, multiplier, offset, tensor_layout, num_threads, device_id, data_dir, crop, rali_cpu = True):
        super(InferencePipe, self).__init__(rali_batch_size, num_threads, device_id, seed=12 + device_id,rali_cpu=rali_cpu)
        #world_size = 1
        #local_rank = 0
        rali_device = 'cpu' if rali_cpu else 'gpu'
        decoder_device = 'cpu' if rali_cpu else 'mixed'
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        
        self.validation_dict = {}
        self.process_validation(image_validation)
        self.set_seed(0)
        self.aug_strength = 0
        self.model_batch_size = model_batch_size
        self.raliMode =  raliMode
        self.data_dir = data_dir
        self.c_img = c_img
        self.h_img = h_img
        self.w_img = w_img
        self.rali_batch_size = rali_batch_size
        self.tensor_dtype = tensor_dtype
        self.multiplier = multiplier
        self.offset = offset
        self.tensor_layout = tensor_layout
        self.reverse_channels = False
        #for tensor output
        self.bs = self.rali_batch_size
        if self.tensor_dtype == types.FLOAT:
            self.out_tensor = np.zeros(( self.bs*self.model_batch_size, self.c_img, int(self.h_img/self.bs), self.w_img,), dtype = "float32")
        elif self.tensor_dtype == types.FLOAT16:
            self.out_tensor = np.zeros(( self.bs*self.model_batch_size, self.c_img, int(self.h_img/self.bs), self.w_img,), dtype = "float16")
        
        self.random_shuffle = True
        self.shard_id = 0
        self.num_shards = 1

        #params for contrast
        self.min_param = self.create_int_param(0)
        self.max_param = self.create_int_param(255)
        #param for brightness
        self.alpha_param = self.create_float_param(1.0)
        self.beta_param = self.create_float_param(10)
        #param for ColorTemperature     
        self.adjustment_param = self.create_int_param(0)
        self.adjustment_param_10 = self.create_int_param(10)
        self.adjustment_param_20 = self.create_int_param(20)
        #param for exposure
        self.shift_param = self.create_float_param(0.0)
        #param for SnPNoise
        self.sdev_param = self.create_float_param(0.0)
        #param for gamma
        self.gamma_shift_param = self.create_float_param(0.0)
        #param for rotate
        self.degree_param = self.create_float_param(0.0)
        self.degree_param_150 = self.create_float_param(150.0)
        self.degree_param_45 = self.create_float_param(45.0)
        #param for lens correction
        self.strength_param = self.create_float_param(1.5)
        self.zoom_param = self.create_float_param(2.0)
        #params for flip
        self.flip_param = self.create_int_param(1)
        #param for snow
        self.snow_param = self.create_float_param(0.1)
        #param for rain
        self.rain_param = self.create_float_param(0.1)
        self.rain_width_param = self.create_int_param(2)
        self.rain_height_param = self.create_int_param(15)
        self.rain_transparency_param = self.create_float_param(0.25)
        #param for blur
        self.blur_param = self.create_int_param(5)
        #param for jitter
        self.kernel_size = self.create_int_param(3)
        #param for warp affine
        self.affine_matrix_param = [0.35,0.25,0.75,1,1,1]
        self.affine_matrix_1_param = [0.5, 0 , 0, 2, 1, 1]
        self.affine_matrix_2_param = [2, 0, 0, 1, 1, 1]
        #param for vignette
        self.vignette_param = self.create_float_param(50)
        #param for blend
        self.blend_param = self.create_float_param(0.5)

        #rali list of augmentation
        self.rali_list = None

        self.decode = ops.ImageDecoder()
        self.resize_img = ops.Resize(resize_x=224, resize_y=224)
        self.warped_img = ops.WarpAffine(matrix=self.affine_matrix_param)
        self.contrast_img = ops.Contrast(min_contrast=self.min_param, max_contrast=self.max_param)
        self.rain_img = ops.Rain(rain=self.rain_param, rain_width = self.rain_width_param, rain_height = self.rain_height_param, rain_transparency =self.rain_transparency_param)
        self.bright_img = ops.Brightness(alpha=self.alpha_param, beta= self.beta_param)
        self.temp_img = ops.ColorTemperature(adjustment_value=self.adjustment_param)
        self.exposed_img = ops.Exposure(exposure=self.shift_param)
        self.vignette_img = ops.Vignette(vignette = self.vignette_param)
        self.blur_img = ops.Blur(blur = self.blur_param)
        self.snow_img = ops.Snow(snow=self.snow_param)
        self.pixelate_img = ops.Pixelate()
        self.snp_img = ops.SnPNoise(snpNoise=self.sdev_param)
        self.gamma_img = ops.GammaCorrection(gamma=self.gamma_shift_param)
        self.rotate_img = ops.Rotate(angle=self.degree_param)
        self.flip_img = ops.Flip(flip=self.flip_param)
        self.blend_img = ops.Blend(blend = self.blend_param)
        if raliMode == 4:
            self.copy_img = ops.Copy()
        elif raliMode == 5:
            self.nop_img = ops.Nop()
        if model_batch_size == 64:
            if raliMode == 1:
                self.rot150_img = ops.Rotate(angle=self.degree_param_150)
                self.flip1_img = ops.Flip(flip=self.flip_param)
                self.rot45_img = ops.Rotate(angle=self.degree_param_45)
            elif raliMode == 2:
                self.warpAffine1_img = ops.WarpAffine(matrix=self.affine_matrix_1_param) #squeeze
                self.fishEye_img = ops.FishEye()
                self.lensCorrection_img = ops.LensCorrection(strength = self.strength_param, zoom = self.zoom_param)
            elif raliMode == 3:
                self.colorTemp1_img = ops.ColorTemperature(adjustment_value=self.adjustment_param_10)
                self.colorTemp2_img = ops.ColorTemperature(adjustment_value=self.adjustment_param_20)
                self.warpAffine2_img = ops.WarpAffine(matrix=self.affine_matrix_2_param) #stretch

    def define_graph(self):
        if self.model_batch_size == 16:
            if self.raliMode == 1:
                self.decode.output = self.decode.rali_c_func_call(self._handle,self.data_dir,self.w_img,self.h_img,self.random_shuffle,self.shard_id,self.num_shards,False)
                self.resize_img.output = self.resize_img.rali_c_func_call(self._handle,self.decode.output,True)
                self.warped_img.output = self.warped_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.contrast_img.output = self.contrast_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.rain_img.output = self.rain_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.bright_img.output = self.bright_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.temp_img.output = self.temp_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.exposed_img.output = self.exposed_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.vignette_img.output = self.vignette_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.blur_img.output = self.blur_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.snow_img.output = self.snow_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.pixelate_img.output = self.pixelate_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.snp_img.output = self.snp_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.gamma_img.output = self.gamma_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.rotate_img.output = self.rotate_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.flip_img.output = self.flip_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.blend_img.output = self.blend_img.rali_c_func_call(self._handle,self.resize_img.output,self.contrast_img.output,True)
            elif self.raliMode == 2:
                self.decode.output = self.decode.rali_c_func_call(self._handle,self.data_dir,self.w_img,self.h_img,self.random_shuffle,self.shard_id,self.num_shards,False)
                self.resize_img.output = self.resize_img.rali_c_func_call(self._handle,self.decode.output,True)
                self.warped_img.output = self.warped_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.contrast_img.output = self.contrast_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.rain_img.output = self.rain_img.rali_c_func_call(self._handle,self.contrast_img.output,True)
                self.bright_img.output = self.bright_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.temp_img.output = self.temp_img.rali_c_func_call(self._handle,self.bright_img.output,True)
                self.exposed_img.output = self.exposed_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.vignette_img.output = self.vignette_img.rali_c_func_call(self._handle,self.exposed_img.output,True)
                self.blur_img.output = self.blur_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.snow_img.output = self.snow_img.rali_c_func_call(self._handle,self.blur_img.output,True)
                self.pixelate_img.output = self.pixelate_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.snp_img.output = self.snp_img.rali_c_func_call(self._handle,self.pixelate_img.output,True)
                self.gamma_img.output = self.gamma_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.rotate_img.output = self.rotate_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.flip_img.output = self.flip_img.rali_c_func_call(self._handle,self.rotate_img.output,True)
                self.blend_img.output = self.blend_img.rali_c_func_call(self._handle,self.resize_img.output,self.contrast_img.output,True)
            elif self.raliMode == 3:
                self.decode.output = self.decode.rali_c_func_call(self._handle,self.data_dir,self.w_img,self.h_img,self.random_shuffle,self.shard_id,self.num_shards,False)
                self.resize_img.output = self.resize_img.rali_c_func_call(self._handle,self.decode.output,True)
                self.warped_img.output = self.warped_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.contrast_img.output = self.contrast_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.rain_img.output = self.rain_img.rali_c_func_call(self._handle,self.warped_img.output,True)
                self.bright_img.output = self.bright_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.temp_img.output = self.temp_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.exposed_img.output = self.exposed_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.vignette_img.output = self.vignette_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.blur_img.output = self.blur_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.snow_img.output = self.snow_img.rali_c_func_call(self._handle,self.vignette_img.output,True)
                self.pixelate_img.output = self.pixelate_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.gamma_img.output = self.gamma_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.snp_img.output = self.snp_img.rali_c_func_call(self._handle,self.gamma_img.output,True)
                self.rotate_img.output = self.rotate_img.rali_c_func_call(self._handle,self.resize_img.output,True)
                self.flip_img.output = self.flip_img.rali_c_func_call(self._handle,self.pixelate_img.output,True)
                self.blend_img.output = self.blend_img.rali_c_func_call(self._handle,self.resize_img.output,self.contrast_img.output,True)
            elif self.raliMode == 4:
                self.decode.output = self.decode.rali_c_func_call(self._handle,self.data_dir,self.w_img,self.h_img,self.random_shuffle,self.shard_id,self.num_shards,False)
                self.resize_img.output = self.resize_img.rali_c_func_call(self._handle,self.decode.output,False)
                for i in range(16):
                    self.copy_img.output = self.copy_img.rali_c_func_call(self._handle,self.resize_img.output,True)
            elif self.raliMode == 5:
                self.decode.output = self.decode.rali_c_func_call(self._handle,self.data_dir,self.w_img,self.h_img,self.random_shuffle,self.shard_id,self.num_shards,False)
                self.resize_img.output = self.resize_img.rali_c_func_call(self._handle,self.decode.output,False)
                for i in range(16):
                    self.nop_img.output = self.nop_img.rali_c_func_call(self._handle,self.resize_img.output,True)
        elif self.model_batch_size == 64:
            if self.raliMode == 1:
                self.decode.output = self.decode.rali_c_func_call(self._handle,self.data_dir,self.w_img,self.h_img,self.random_shuffle,self.shard_id,self.num_shards,False)
                self.resize_img.output = self.resize_img.rali_c_func_call(self._handle,self.decode.output,False)
                self.setof16_mode1(self.resize_img.output)
                self.rot150_img.output = self.rot150_img.rali_c_func_call(self._handle,self.resize_img.output,False)
                self.setof16_mode1(self.rot150_img.output)
                self.flip1_img.output = self.flip1_img.rali_c_func_call(self._handle,self.resize_img.output,False)
                self.setof16_mode1(self.flip1_img.output)
                self.rot45_img.output = self.rot45_img.rali_c_func_call(self._handle,self.resize_img.output,False)
                self.setof16_mode1(self.rot45_img.output)
            elif self.raliMode == 2:
                self.decode.output = self.decode.rali_c_func_call(self._handle,self.data_dir,self.w_img,self.h_img,self.random_shuffle,self.shard_id,self.num_shards,False)
                self.resize_img.output = self.resize_img.rali_c_func_call(self._handle,self.decode.output,False)
                self.setof16_mode1(self.resize_img.output)
                self.warpAffine1_img.output = self.warpAffine1_img.rali_c_func_call(self._handle,self.resize_img.output,False)
                self.setof16_mode1(self.warpAffine1_img.output)
                self.fishEye_img.output = self.fishEye_img.rali_c_func_call(self._handle,self.resize_img.output,False)
                self.setof16_mode1(self.fishEye_img.output)
                self.lensCorrection_img.output = self.lensCorrection_img.rali_c_func_call(self._handle,self.resize_img.output,False)
                self.setof16_mode1(self.lensCorrection_img.output)
            elif self.raliMode == 3:
                self.decode.output = self.decode.rali_c_func_call(self._handle,self.data_dir,self.w_img,self.h_img,self.random_shuffle,self.shard_id,self.num_shards,False)
                self.resize_img.output = self.resize_img.rali_c_func_call(self._handle,self.decode.output,False)
                self.setof16_mode1(self.resize_img.output)
                self.colorTemp1_img.output = self.colorTemp1_img.rali_c_func_call(self._handle,self.resize_img.output,False)
                self.setof16_mode1(self.colorTemp1_img.output)
                self.colorTemp2_img.output = self.colorTemp2_img.rali_c_func_call(self._handle,self.resize_img.output,False)
                self.setof16_mode1(self.colorTemp2_img.output)
                self.warpAffine2_img.output = self.warpAffine2_img.rali_c_func_call(self._handle,self.resize_img.output,False)
                self.setof16_mode1(self.warpAffine2_img.output)
            elif self.raliMode == 4:
                self.decode.output = self.decode.rali_c_func_call(self._handle,self.data_dir,self.w_img,self.h_img,self.random_shuffle,self.shard_id,self.num_shards,False)
                self.resize_img.output = self.resize_img.rali_c_func_call(self._handle,self.decode.output,False)
                for i in range(64):
                    self.copy_img.output = self.copy_img.rali_c_func_call(self._handle,self.resize_img.output,True)
            elif self.raliMode == 5:
                self.decode.output = self.decode.rali_c_func_call(self._handle,self.data_dir,self.w_img,self.h_img,self.random_shuffle,self.shard_id,self.num_shards,False)
                self.resize_img.output = self.resize_img.rali_c_func_call(self._handle,self.decode.output,False)
                for i in range(64):
                    self.nop_img.output = self.nop_img.rali_c_func_call(self._handle,self.resize_img.output,True)

    def get_input_name(self):
        self.img_names_length = np.empty(self.rali_batch_size, dtype="int32")
        self.img_names_size = self.GetImageNameLen(self.img_names_length)
        # Images names of a batch
        self.Img_name = self.GetImageName(self.img_names_size)
        return self.Img_name.decode()

    def process_validation(self, validation_list):
        for i in range(len(validation_list)):
            name, groundTruthIndex = validation_list[i].split(' ')
            self.validation_dict[name] = groundTruthIndex

    def get_ground_truth(self):
        return self.validation_dict[self.get_input_name()]

    def setof16_mode1(self, input_image):
        self.resize_img.output = self.resize_img.rali_c_func_call(self._handle,input_image,True)
        self.warped_img.output = self.warped_img.rali_c_func_call(self._handle,input_image,True)
        self.contrast_img.output = self.contrast_img.rali_c_func_call(self._handle,input_image,True)
        self.rain_img.output = self.rain_img.rali_c_func_call(self._handle,input_image,True)
        self.bright_img.output = self.bright_img.rali_c_func_call(self._handle,input_image,True)
        self.temp_img.output = self.temp_img.rali_c_func_call(self._handle,input_image,True)
        self.exposed_img.output = self.exposed_img.rali_c_func_call(self._handle,input_image,True)
        self.vignette_img.output = self.vignette_img.rali_c_func_call(self._handle,input_image,True)
        self.blur_img.output = self.blur_img.rali_c_func_call(self._handle,input_image,True)
        self.snow_img.output = self.snow_img.rali_c_func_call(self._handle,input_image,True)
        self.pixelate_img.output = self.pixelate_img.rali_c_func_call(self._handle,input_image,True)
        self.snp_img.output = self.snp_img.rali_c_func_call(self._handle,input_image,True)
        self.gamma_img.output = self.gamma_img.rali_c_func_call(self._handle,input_image,True)
        self.rotate_img.output = self.rotate_img.rali_c_func_call(self._handle,input_image,True)
        self.flip_img.output = self.flip_img.rali_c_func_call(self._handle,input_image,True)
        self.blend_img.output = self.blend_img.rali_c_func_call(self._handle,input_image,self.contrast_img.output,True)

    def updateAugmentationParameter(self, augmentation):
        #values for contrast
        self.aug_strength = augmentation
        min = int(augmentation*100)
        max = 150 + int((1-augmentation)*100)
        self.update_int_param(min, self.min_param)
        self.update_int_param(max, self.max_param)

        #values for brightness
        alpha = augmentation*1.95
        self.update_float_param(alpha, self.alpha_param)

        #values for ColorTemperature
        adjustment = (augmentation*99) if ((int(augmentation*100)) % 2 == 0) else (-1*augmentation*99)
        adjustment = int(adjustment)
        self.update_int_param(adjustment, self.adjustment_param)

        #values for exposure
        shift = augmentation*0.95
        self.update_float_param(shift, self.shift_param)

        #values for SnPNoise
        sdev = augmentation*0.7
        self.update_float_param(sdev, self.sdev_param)

        #values for gamma
        gamma_shift = augmentation*5.0
        self.update_float_param(gamma_shift, self.gamma_shift_param)

    def renew_parameters(self):
        curr_degree = self.get_float_value(self.degree_param)
        #values for rotation change
        degree = self.aug_strength * 100
        self.update_float_param(curr_degree+degree, self.degree_param)

    def start_iterator(self):
        #self.reset()
        self.raliResetLoaders()

    def get_next_augmentation(self, imageIterator):
        if self.isEmpty() == 1:
            return -1
            #raise StopIteration
        self.renew_parameters()
        self.out_image = imageIterator.next()
        # print (self.out_tensor.shape)
        # if(types.NCHW == self.tensor_layout):
        #     self.copyToTensorNCHW(self.out_tensor, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))
        # else:
        #     self.copyToTensorNHWC(self.out_tensor, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))
        return self.out_image, self.out_tensor

    def get_rali_list(self, raliMode, model_batch_size):
        if model_batch_size == 16:
            if raliMode == 1:
                self.rali_list = raliList_mode1_16
            elif raliMode == 2:
                self.rali_list = raliList_mode2_16
            elif raliMode == 3:
                self.rali_list = raliList_mode3_16
            elif raliMode == 4:
                self.rali_list = raliList_mode4_16
            elif raliMode == 5:
                self.rali_list = raliList_mode5_16
        elif model_batch_size == 64:
            if raliMode == 1:
                self.rali_list = raliList_mode1_64
            elif raliMode == 2:
                self.rali_list = raliList_mode2_64
            elif raliMode == 3:
                self.rali_list = raliList_mode3_64
            elif raliMode == 4:
                self.rali_list = raliList_mode4_64
            elif raliMode == 5:
                self.rali_list = raliList_mode5_64
        return self.rali_list
