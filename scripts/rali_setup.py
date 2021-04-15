import sys
#setup python path for RALI
sys.path.append('/opt/rocm/mivisionx/rali/python/')
from rali import *
from rali_image_iterator import *
from rali_common import *
from enum import Enum
from amd.rali.plugin.pytorch import RALIClassificationIterator
from amd.rali.plugin.pytorch import RALI_iterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types

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
#class DataLoader(RaliGraph):
class InferencePipe(Pipeline):
	#def __init__(self, input_path, rali_batch_size, model_batch_size, input_color_format, affinity, image_validation, h_img, w_img, raliMode, loop_parameter,
	#			 tensor_layout = TensorLayout.NCHW, reverse_channels = False, multiplier = [1.0,1.0,1.0], offset = [0.0, 0.0, 0.0], tensor_dtype=TensorDataType.FLOAT32):
	#	RaliGraph.__init__(self, rali_batch_size, affinity)
	def __init__(self, image_validation, model_batch_size, raliMode, h_img, w_img, rali_batch_size, tensor_layout, num_threads, device_id, data_dir, crop, rali_cpu = True):
		super(InferencePipe, self).__init__(rali_batch_size, num_threads, device_id, seed=12 + device_id,rali_cpu=rali_cpu)
		world_size = 1
		local_rank = 0
		self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
		rali_device = 'cpu' if rali_cpu else 'gpu'
		decoder_device = 'cpu' if rali_cpu else 'mixed'
		device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
		host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
		self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB,
													device_memory_padding=device_memory_padding,
													host_memory_padding=host_memory_padding)
		self.validation_dict = {}
		self.process_validation(image_validation)
		#self.setSeed(0)
		self.aug_strength = 0
		#params for contrast
		self.min_param = RaliIntParameter(0)
		self.max_param = RaliIntParameter(255)
		#param for brightness
		self.alpha_param = RaliFloatParameter(0.0)
		#param for colorTemp		
		self.adjustment_param = RaliIntParameter(0)
		#param for exposure
		self.shift_param = RaliFloatParameter(0.0)
		#param for SnPNoise
		self.sdev_param = RaliFloatParameter(0.0)
		#param for gamma
		self.gamma_shift_param = RaliFloatParameter(0.0)
		#param for rotate
		self.degree_param = RaliFloatParameter(0.0)

		#rali list of augmentation
		self.rali_list = None

        self.resize = ops.Resize(resize_x=h_img, resize_y=w_img, preserve = True)	
        self.warped = ops.WarpAffine(preserve=True)
        self.contrast_img = ops.Contrast(min_contrast=self.min_param, max_contrast=self.max_param, preserve=True)
        self.rain_img = ops.Rain(preserve=True)
        self.bright_img = ops.Brightness(self.alpha_param, preserve=True)
        self.temp_img = ops.ColorTemp(alpha=self.adjustment_param, preserve=True)
        self.exposed_img = ops.Exposure(exposure=self.shift_param, preserve=True)
        self.vignette_img = ops.Vignette(preserve=True)
        self.blur_img = ops.Blur(preserve=True)
        self.snow_img = ops.Snow(preserve=True)
        self.pixelate_img = ops.Pixelate(preserve=True)
        self.snp_img = ops.SnPNoise(snpNoise=self.sdev_param, preserve=True)
        self.gamma_img = ops.GammaCorrection(gamma=self.gamma_shift_param, preserve=True)
        self.rotate_img = ops.Rotate(angle=self.degree_param, preserve=True)
        self.flip_img = ops.Flip(self.input, True, 1)
        self.blend_img = ops.Blend(self.contrast_img, preserve=True)

        """
		if model_batch_size == 16:
			if raliMode == 1:
				#self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				self.resize = ops.Resize(resize_x=h_img, resize_y=w_img, preserve = True)
				
				self.warped = ops.WarpAffine(preserve=True)

				self.contrast_img = ops.Contrast(min_contrast=self.min_param, max_contrast=self.max_param, preserve=True)
				self.rain_img = ops.Rain(preserve=True)

				self.bright_img = ops.Brightness(self.alpha_param, preserve=True)
				self.temp_img = ops.ColorTemp(alpha=self.adjustment_param, preserve=True)

				self.exposed_img = ops.Exposure(exposure=self.shift_param, preserve=True)
				self.vignette_img = ops.Vignette(preserve=True)
				self.blur_img = ops.Blur(preserve=True)
				self.snow_img = ops.Snow(preserve=True)

				self.pixelate_img = ops.Pixelate(preserve=True)
				self.snp_img = ops.SnPNoise(snpNoise=self.sdev_param, preserve=True)
				self.gamma_img = ops.GammaCorrection(gamma=self.gamma_shift_param, preserve=True)

				self.rotate_img = ops.Rotate(angle=self.degree_param, preserve=True)

				self.flip_img = ops.Flip(self.input, True, 1)
				#self.jitter_img = self.jitter(self.input, True)
				
				self.blend_img = ops.Blend(self.contrast_img, preserve=True)
				
			elif raliMode == 2:
				#self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				self.resize = ops.Resize(resize_x=h_img, resize_y=w_img, preserve = True)
				
				self.warped = ops.WarpAffine(preserve=True)

				self.contrast_img = ops.Contrast(min_contrast=self.min_param, max_contrast=self.max_param, preserve=True)
				self.rain_img = ops.Rain(preserve=True)

				self.bright_img = ops.Brightness(self.alpha_param, preserve=True)
				self.temp_img = ops.ColorTemp(alpha=self.adjustment_param, preserve=True)

				self.exposed_img = ops.Exposure(exposure=self.shift_param, preserve=True)
				self.vignette_img = ops.Vignette(preserve=True)
				self.blur_img = ops.Blur(preserve=True)
				self.snow_img = ops.Snow(preserve=True)

				self.pixelate_img = ops.Pixelate(preserve=True)
				self.snp_img = ops.SnPNoise(snpNoise=self.sdev_param, preserve=True)
				self.gamma_img = ops.GammaCorrection(gamma=self.gamma_shift_param, preserve=True)

				self.rotate_img = ops.Rotate(angle=self.degree_param, preserve=True)
				self.flip_img = ops.Flip(self.input, True, 1)
				#self.jitter_img = self.jitter(self.input, True)
				
				self.blend_img = ops.Blend(self.contrast_img, preserve=True)
			elif raliMode == 3:
				#self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				self.resize = ops.Resize(device=rali_device, resize_x=h_img, resize_y=w_img)
				
				self.warped = ops.WarpAffine(preserve=True)

				self.contrast_img = ops.Contrast(min_contrast=self.min_param, max_contrast=self.max_param, preserve=True)
				self.rain_img = ops.Rain(preserve=True)

				self.bright_img = ops.Brightness(self.alpha_param, preserve=True)
				self.temp_img = ops.ColorTemp(alpha=self.adjustment_param, preserve=True)

				self.exposed_img = ops.Exposure(exposure=self.shift_param, preserve=True)
				self.vignette_img = ops.Vignette(preserve=True)
				self.blur_img = ops.Blur(preserve=True)
				self.snow_img = ops.Snow(preserve=True)

				self.pixelate_img = ops.Pixelate(preserve=True)
				self.snp_img = ops.SnPNoise(snpNoise=self.sdev_param, preserve=True)
				self.gamma_img = ops.GammaCorrection(gamma=self.gamma_shift_param, preserve=True)

				self.rotate_img = ops.Rotate(angle=self.degree_param, preserve=True)
				self.flip_img = ops.Flip(self.input, True, 1)
				#self.jitter_img = self.jitter(self.input, True)
				
				self.blend_img = ops.Blend(self.contrast_img, preserve=True)
			elif raliMode == 4:
				#self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				self.resize = ops.Resize(device=rali_device, resize_x=h_img, resize_y=w_img)
				
				#for i in range(15):
				#	self.copy_img = ops.Copy(preserve=True)
				
			elif raliMode == 5:
				#self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				self.resize = ops.Resize(device=rali_device, resize_x=h_img, resize_y=w_img)
				
				#for i in range(15):
				#	self.nop_img = self.Nop(preserve=True)
				
		elif model_batch_size == 64:
			if raliMode == 1:
				#self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				#self.input = self.resize(self.jpg_img, h_img, w_img, False)
				self.resize = ops.Resize(device=rali_device, resize_x=h_img, resize_y=w_img)
				
				self.rot150_img = ops.Rotate(preserve=False, angle=150)
				self.flip_img = ops.Flip(preserve=False)
				self.rot45_img = ops.Rotate(preserve=False, angle=45)

				self.setof16_mode1(self.input, h_img, w_img)
				self.setof16_mode1(self.rot45_img, h_img, w_img)
				self.setof16_mode1(self.flip_img, h_img, w_img)
				self.setof16_mode1(self.rot150_img , h_img, w_img)
				
			elif raliMode == 2:
				#self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				#self.input = self.resize(self.jpg_img, h_img, w_img, False)
				self.resize = ops.Resize(device=rali_device, resize_x=h_img, resize_y=w_img)
				
				#self.warpAffine2_img = self.warpAffine(self.input, False, [[1.5,0],[0,1],[None,None]])
				self.warpAffine1_img = self.warpAffine(self.input, False, [[0.5,0],[0,2],[None,None]]) #squeeze
				self.fishEye_img = self.fishEye(self.input, False)
				self.lensCorrection_img = self.lensCorrection(self.input, False, 1.5, 2)

				self.setof16_mode1(self.input, h_img, w_img)
				self.setof16_mode1(self.warpAffine1_img, h_img, w_img)
				self.setof16_mode1(self.fishEye_img, h_img, w_img)
				self.setof16_mode1(self.lensCorrection_img, h_img, w_img)
				
			elif raliMode == 3:
				#self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				#self.input = self.resize(self.jpg_img, h_img, w_img, False)
				self.resize = ops.Resize(device=rali_device, resize_x=h_img, resize_y=w_img)
				
				self.colorTemp1_img = self.colorTemp(self.input, False, 10)
				self.colorTemp2_img = self.colorTemp(self.input, False, 20)
				self.warpAffine2_img = self.warpAffine(self.input, False, [[2,0],[0,1],[None,None]]) #stretch

				self.setof16_mode1(self.input, h_img, w_img)
				self.setof16_mode1(self.colorTemp1_img, h_img, w_img)
				self.setof16_mode1(self.colorTemp2_img, h_img, w_img)
				self.setof16_mode1(self.warpAffine2_img , h_img, w_img)
				
			elif raliMode == 4:
				#self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				#self.input = self.resize(self.jpg_img, h_img, w_img, True)
				self.resize = ops.Resize(device=rali_device, resize_x=h_img, resize_y=w_img)
				
				for i in range(63):
					self.copy_img = self.copy(self.input, True)
				
			elif raliMode == 5:	
				#self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				#self.input = self.resize(self.jpg_img, h_img, w_img, True)
				self.resize = ops.Resize(device=rali_device, resize_x=h_img, resize_y=w_img)
				
				for i in range(63):
					self.nop_img = self.nop(self.input, True)
		"""		
		#rali iterator
		#if self.build() != 0:
		#	raise Exception('Failed to build the augmentation graph')
		"""
		self.tensor_format = tensor_layout
		self.multiplier = multiplier
		self.offset = offset
		self.reverse_channels = reverse_channels
		self.tensor_dtype = tensor_dtype
		self.w = self.getOutputWidth()
		self.h = self.getOutputHeight()
		self.b = self.getBatchSize()
		self.n = self.raliGetAugmentationBranchCount()
		color_format = self.getOutputColorFormat()
		self.p = (1 if color_format is ColorFormat.IMAGE_U8 else 3)
		height = self.h*self.n
		self.out_image = np.zeros((height, self.w, self.p), dtype = "uint8")
		self.out_tensor = np.zeros(( self.b*self.n, self.p, self.h/self.b, self.w,), dtype = "float32")
		"""
	def define_graph(self, model_batch_size, raliMode):
		#rng = self.coin()
		self.jpegs, self.labels = self.input(name="Reader")
		images = self.decode(self.jpegs)

        if model_batch_size == 16:
            if raliMode == 1:
                images = self.resize(images)
                images = self.warped(images)
                images = self.contrast(images)
                images = self.rain_img(images)
                images = self.bright_img(images)
                images = self.temp_img(images)
                images = self.exposed_img(images)
                images = self.vignette_img(images)
                images = self.blur_img(images)
                images = self.snow_img(images)
                images = self.pixelate_img(images)
                images = self.snp_img(images)
                images = self.gamma_img(images)
                images = self.rotate_img(images)
                images = self.flip_img(images)
                output = self.blend_img(images)
                return [output, self.labels]
            elif raliMode == 2:
                

	def get_input_name(self):
		size = self.raliGetImageNameLen(0)
		ret = ctypes.create_string_buffer(size)
		self.raliGetImageName(ret, 0)
		return ret.value

	def process_validation(self, validation_list):
		for i in range(len(validation_list)):
			name, groundTruthIndex = validation_list[i].split(' ')
			self.validation_dict[name] = groundTruthIndex

	def get_ground_truth(self):
		return self.validation_dict[self.get_input_name()]

	def setof16_mode1(self, input_image, h_img, w_img):
		self.resized_image = ops.Resize(device=rali_device, resize_x=h_img, resize_y=w_img)

		self.warped = ops.WarpAffine(preserve=True)

		self.contrast_img = ops.Contrast(min_contrast=self.min_param, max_contrast=self.max_param, preserve=True)
		self.rain_img = ops.Rain(preserve=True)

		self.bright_img = ops.Brightness(self.alpha_param, preserve=True)
		self.temp_img = ops.ColorTemp(alpha=self.adjustment_param, preserve=True)

		self.exposed_img = ops.Exposure(exposure=self.shift_param, preserve=True)
		self.vignette_img = ops.Vignette(preserve=True)
		self.blur_img = ops.Blur(preserve=True)
		self.snow_img = ops.Snow(preserve=True)

		self.pixelate_img = ops.Pixelate(preserve=True)
		self.snp_img = ops.SnPNoise(snpNoise=self.sdev_param, preserve=True)
		self.gamma_img = ops.GammaCorrection(gamma=self.gamma_shift_param, preserve=True)

		self.rotate_img = ops.Rotate(angle=self.degree_param, preserve=True)
		self.flip_img = ops.Flip(self.input, True, 1)
		#self.jitter_img = self.jitter(self.input, True)
				
		self.blend_img = ops.Blend(self.contrast_img, preserve=True)

	def updateAugmentationParameter(self, augmentation):
		#values for contrast
		self.aug_strength = augmentation
		min = int(augmentation*100)
		max = 150 + int((1-augmentation)*100)
		self.min_param.update(min)
		self.max_param.update(max)

		#values for brightness
		alpha = augmentation*1.95
		self.alpha_param.update(alpha)

		#values for colorTemp
		adjustment = (augmentation*99) if ((int(augmentation*100)) % 2 == 0) else (-1*augmentation*99)
		adjustment = int(adjustment)
		self.adjustment_param.update(adjustment)

		#values for exposure
		shift = augmentation*0.95
		self.shift_param.update(shift)

		#values for SnPNoise
		sdev = augmentation*0.7
		self.sdev_param.update(sdev)

		#values for gamma
		gamma_shift = augmentation*5.0
		self.gamma_shift_param.update(gamma_shift)



	def renew_parameters(self):
		curr_degree = self.degree_param.get()
		#values for rotation change
		degree = self.aug_strength * 100
		self.degree_param.update(curr_degree+degree)

	def start_iterator(self):
		#self.reset()
		self.raliResetLoaders()
		
	def get_next_augmentation(self):
		if self.IsEmpty() == 1:
			return -1
			#raise StopIteration
		self.renew_parameters()
		"""
		if self.run() != 0:
			#raise StopIteration
			return -1
		self.copyToNPArray(self.out_image)
		if(TensorLayout.NCHW == self.tensor_format):
			self.copyToTensorNCHW(self.out_tensor, self.multiplier, self.offset, self.reverse_channels, self.tensor_dtype)

		else:
			self.copyToTensorNHWC(self.out_tensor, self.multiplier, self.offset, self.reverse_channels, self.tensor_dtype)
		"""
		self.out_image , self.out_tensor = self.next()
		return self.out_image , self.out_tensor

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
