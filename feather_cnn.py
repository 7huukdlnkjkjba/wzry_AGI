import ctypes
import os
import numpy as np
import time


class FeatherCNN:
    """
    FeatherCNN推理接口 - 腾讯开源的快速推断库
    GitHub: https://github.com/Tencent/FeatherCNN
    
    特性：
    - 支持ONNX模型加载
    - 高性能CPU推理
    - 支持INT8量化加速
    - 多线程推理支持
    
    编译FeatherCNN：
    1. git clone https://github.com/Tencent/FeatherCNN.git
    2. cd FeatherCNN
    3. mkdir build && cd build
    4. cmake .. -DUSE_OPENMP=ON
    5. make -j4
    6. 生成的库文件在 build/lib/ 目录下
    """
    
    def __init__(self):
        self.initialized = False
        self.model_loaded = False
        self.feather_lib = None
        self.feather_handle = None
        
        # 尝试加载FeatherCNN库
        self._load_library()
    
    def _load_library(self):
        """加载FeatherCNN共享库"""
        # 尝试查找FeatherCNN库（支持多种平台）
        lib_paths = []
        
        # Windows
        if os.name == 'nt':
            lib_paths.extend([
                'FeatherCNN.dll',
                './FeatherCNN.dll',
                './lib/FeatherCNN.dll',
                './build/lib/FeatherCNN.dll',
                './FeatherCNN/build/lib/FeatherCNN.dll',
                os.path.expanduser('~/FeatherCNN/build/lib/FeatherCNN.dll')
            ])
        # Linux/Mac
        else:
            lib_paths.extend([
                './libfeathercnn.so',
                './build/lib/libfeathercnn.so',
                './FeatherCNN/build/lib/libfeathercnn.so',
                os.path.expanduser('~/FeatherCNN/build/lib/libfeathercnn.so'),
                '/usr/local/lib/libfeathercnn.so',
                '/usr/lib/libfeathercnn.so'
            ])
        
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                try:
                    self.feather_lib = ctypes.CDLL(lib_path)
                    self.initialized = True
                    print(f"[FeatherCNN] Loaded library from: {lib_path}")
                    self._setup_api()
                    return
                except Exception as e:
                    print(f"[FeatherCNN] Failed to load {lib_path}: {e}")
        
        # 如果找不到FeatherCNN库，使用PyTorch作为备选
        print("[FeatherCNN] FeatherCNN library not found, using PyTorch fallback")
        print("[FeatherCNN] To use FeatherCNN acceleration:")
        print("  1. Clone: git clone https://github.com/Tencent/FeatherCNN.git")
        print("  2. Build: cd FeatherCNN && mkdir build && cd build && cmake .. -DUSE_OPENMP=ON && make -j4")
        print("  3. Copy: cp lib/FeatherCNN.dll (or libfeathercnn.so) to project root")
        self.initialized = True  # 标记为已初始化（使用fallback）
    
    def _setup_api(self):
        """设置FeatherCNN API函数签名"""
        if self.feather_lib is None:
            return
        
        try:
            # 创建FeatherCNN实例
            self.feather_lib.FeatherCNN_new.restype = ctypes.c_void_p
            self.feather_lib.FeatherCNN_new.argtypes = []
            
            # 销毁FeatherCNN实例
            self.feather_lib.FeatherCNN_delete.argtypes = [ctypes.c_void_p]
            
            # 加载模型
            self.feather_lib.FeatherCNN_loadModel.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.feather_lib.FeatherCNN_loadModel.restype = ctypes.c_int
            
            # 设置输入形状
            self.feather_lib.FeatherCNN_setInputShape.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            
            # 初始化推理引擎
            self.feather_lib.FeatherCNN_init.argtypes = [ctypes.c_void_p]
            self.feather_lib.FeatherCNN_init.restype = ctypes.c_int
            
            # 前向推理
            self.feather_lib.FeatherCNN_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
            self.feather_lib.FeatherCNN_forward.restype = ctypes.c_int
            
            # 获取输出数量
            self.feather_lib.FeatherCNN_getOutputNum.argtypes = [ctypes.c_void_p]
            self.feather_lib.FeatherCNN_getOutputNum.restype = ctypes.c_int
            
            # 获取输出形状
            self.feather_lib.FeatherCNN_getOutputShape.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
            ]
            
            # 获取输出数据
            self.feather_lib.FeatherCNN_getOutput.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.feather_lib.FeatherCNN_getOutput.restype = ctypes.POINTER(ctypes.c_float)
            
            # 设置线程数
            self.feather_lib.FeatherCNN_setThreads.argtypes = [ctypes.c_void_p, ctypes.c_int]
            
            print("[FeatherCNN] API setup completed")
            
        except AttributeError as e:
            print(f"[FeatherCNN] API setup failed: {e}")
            print("[FeatherCNN] This FeatherCNN library may not have the expected API")
            self.feather_lib = None
    
    def load_model(self, model_path, input_shape=(1, 3, 640, 640)):
        """
        加载ONNX模型
        :param model_path: ONNX模型路径
        :param input_shape: 输入张量形状 (batch, channels, height, width)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.input_shape = input_shape
        
        if self.feather_lib is not None:
            # 使用FeatherCNN加载模型
            self._load_feather_model(model_path, input_shape)
        else:
            # 使用PyTorch加载ONNX模型作为备选
            self._load_pytorch_fallback(model_path)
        
        self.model_loaded = True
        print(f"[FeatherCNN] Model loaded: {model_path}")
    
    def _load_feather_model(self, model_path, input_shape):
        """使用FeatherCNN加载模型"""
        # 创建FeatherCNN实例
        self.feather_handle = self.feather_lib.FeatherCNN_new()
        
        # 加载ONNX模型
        result = self.feather_lib.FeatherCNN_loadModel(
            self.feather_handle,
            model_path.encode('utf-8')
        )
        
        if result != 0:
            self.feather_lib.FeatherCNN_delete(self.feather_handle)
            raise RuntimeError(f"Failed to load model with FeatherCNN (error code: {result})")
        
        # 设置输入形状
        self.feather_lib.FeatherCNN_setInputShape(
            self.feather_handle,
            input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        )
        
        # 初始化推理引擎
        result = self.feather_lib.FeatherCNN_init(self.feather_handle)
        if result != 0:
            self.feather_lib.FeatherCNN_delete(self.feather_handle)
            raise RuntimeError(f"Failed to initialize FeatherCNN (error code: {result})")
        
        print(f"[FeatherCNN] Model loaded successfully with input shape: {input_shape}")
    
    def _load_pytorch_fallback(self, model_path):
        """使用PyTorch加载ONNX模型作为备选"""
        try:
            import torch
            import onnxruntime as ort
            
            # 使用ONNX Runtime作为备选（比PyTorch更快）
            print("[FeatherCNN] Using ONNX Runtime as fallback")
            self.ort_session = ort.InferenceSession(model_path)
            self.use_onnxruntime = True
            
        except ImportError:
            # 如果没有ONNX Runtime，使用PyTorch
            import torch
            print("[FeatherCNN] Using PyTorch as fallback")
            self.torch_model = torch.onnx.load(model_path)
            self.torch_device = torch.device('cpu')
            self.use_onnxruntime = False
    
    def infer(self, input_data):
        """
        执行推理
        :param input_data: 输入数据 (numpy array 或 torch tensor)
        :return: 推理结果列表
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # 确保输入数据是numpy array
        if isinstance(input_data, np.ndarray):
            input_array = input_data
        else:
            input_array = np.array(input_data)
        
        # 确保输入形状正确
        if len(input_array.shape) == 3:  # (H, W, C)
            input_array = input_array.transpose(2, 0, 1)  # (C, H, W)
            input_array = np.expand_dims(input_array, 0)  # (1, C, H, W)
        
        if self.feather_lib is not None:
            return self._infer_feather(input_array)
        elif hasattr(self, 'use_onnxruntime') and self.use_onnxruntime:
            return self._infer_onnxruntime(input_array)
        else:
            return self._infer_pytorch_fallback(input_array)
    
    def _infer_feather(self, input_array):
        """使用FeatherCNN执行推理"""
        # 转换为float32
        input_data = input_array.astype(np.float32)
        
        # 获取输入指针
        input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # 执行推理
        result = self.feather_lib.FeatherCNN_forward(self.feather_handle, input_ptr)
        if result != 0:
            raise RuntimeError(f"FeatherCNN forward failed (error code: {result})")
        
        # 获取输出数量
        output_num = self.feather_lib.FeatherCNN_getOutputNum(self.feather_handle)
        
        # 获取每个输出
        results = []
        for i in range(output_num):
            # 获取输出形状
            dims = (ctypes.c_int * 4)()
            self.feather_lib.FeatherCNN_getOutputShape(self.feather_handle, i, dims)
            
            # 计算输出大小
            output_size = dims[0] * dims[1] * dims[2] * dims[3]
            
            # 获取输出数据
            output_ptr = self.feather_lib.FeatherCNN_getOutput(self.feather_handle, i)
            
            # 转换为numpy数组
            output_data = np.ctypeslib.as_array(
                (ctypes.c_float * output_size).from_address(ctypes.addressof(output_ptr.contents))
            )
            results.append(output_data.reshape(dims[0], dims[1], dims[2], dims[3]).copy())
        
        return results
    
    def _infer_onnxruntime(self, input_array):
        """使用ONNX Runtime执行推理"""
        input_data = input_array.astype(np.float32)
        
        # 执行推理
        outputs = self.ort_session.run(None, {'input': input_data})
        
        # 转换为numpy数组
        return [output for output in outputs]
    
    def _infer_pytorch_fallback(self, input_array):
        """使用PyTorch执行推理作为备选"""
        import torch
        
        input_tensor = torch.from_numpy(input_array).float().to(self.torch_device)
        
        # 执行推理
        with torch.no_grad():
            outputs = self.torch_model(input_tensor)
        
        # 转换为numpy数组
        if isinstance(outputs, torch.Tensor):
            return [outputs.cpu().numpy()]
        elif isinstance(outputs, (list, tuple)):
            return [o.cpu().numpy() for o in outputs]
        else:
            return [np.array(outputs)]
    
    def benchmark(self, iterations=100):
        """
        性能基准测试
        :param iterations: 迭代次数
        :return: 平均推理时间（毫秒）
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # 创建随机输入
        input_data = np.random.randn(*self.input_shape).astype(np.float32)
        
        # 预热
        for _ in range(10):
            self.infer(input_data)
        
        # 正式测试
        start_time = time.time()
        for _ in range(iterations):
            self.infer(input_data)
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) / iterations * 1000
        print(f"[FeatherCNN] Benchmark: {avg_time_ms:.2f} ms per inference ({1000/avg_time_ms:.1f} FPS)")
        return avg_time_ms
    
    def set_threads(self, num_threads):
        """设置推理线程数"""
        if self.feather_lib is not None:
            self.feather_lib.FeatherCNN_setThreads(self.feather_handle, num_threads)
            print(f"[FeatherCNN] Set threads: {num_threads}")
        else:
            print(f"[FeatherCNN] Cannot set threads (using fallback)")
    
    def __del__(self):
        """释放资源"""
        if self.feather_lib is not None and self.feather_handle is not None:
            self.feather_lib.FeatherCNN_delete(self.feather_handle)


class FeatherCNNManager:
    """
    FeatherCNN管理器 - 管理多个模型的推理
    """
    
    _instance = None
    
    def __init__(self):
        self.models = {}
        self.default_threads = 1
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = FeatherCNNManager()
        return cls._instance
    
    def load_model(self, model_name, model_path, input_shape=(1, 3, 640, 640)):
        """加载模型"""
        model = FeatherCNN()
        model.load_model(model_path, input_shape)
        model.set_threads(self.default_threads)
        self.models[model_name] = model
        return model
    
    def get_model(self, model_name):
        """获取模型"""
        return self.models.get(model_name)
    
    def infer(self, model_name, input_data):
        """执行推理"""
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model not found: {model_name}")
        return model.infer(input_data)
    
    def set_default_threads(self, num_threads):
        """设置默认线程数"""
        self.default_threads = num_threads
        for model in self.models.values():
            model.set_threads(num_threads)
    
    def benchmark_all(self):
        """对所有模型进行基准测试"""
        for name, model in self.models.items():
            print(f"[FeatherCNN Manager] Benchmarking {name}...")
            model.benchmark()