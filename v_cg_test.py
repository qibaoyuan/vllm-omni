# import torch
# import numpy as np

# def compare_pt_files(file1_path: str, file2_path: str, rtol: float = 1e-5, atol: float = 1e-8):
#     """
#     对比两个.pt文件中的tensor值是否完全一样
    
#     Args:
#         file1_path: 第一个.pt文件路径
#         file2_path: 第二个.pt文件路径
#         rtol: 相对误差容忍度（用于近似比较）
#         atol: 绝对误差容忍度（用于近似比较）
    
#     Returns:
#         bool: 是否完全相等
#     """
#     # 加载两个文件
#     print(f"Loading file 1: {file1_path}")
#     data1 = torch.load(file1_path, map_location='cuda').float()
    
#     print(f"Loading file 2: {file2_path}")
#     data2 = torch.load(file2_path, map_location='cuda').float()
    
#     # 检查类型
#     print(f"\nFile 1 type: {type(data1)}")
#     print(f"File 2 type: {type(data2)}")
    
#     # 如果都是tensor，直接比较
#     if isinstance(data1, torch.Tensor) and isinstance(data2, torch.Tensor):
#         return compare_tensors(data1, data2, rtol, atol)
    
#     # 如果是dict，递归比较
#     elif isinstance(data1, dict) and isinstance(data2, dict):
#         return compare_dicts(data1, data2, rtol, atol)
    
#     # 如果是list，逐个比较
#     elif isinstance(data1, (list, tuple)) and isinstance(data2, (list, tuple)):
#         return compare_lists(data1, data2, rtol, atol)
    
#     else:
#         print(f"Unsupported types: {type(data1)} vs {type(data2)}")
#         return False


# def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol: float, atol: float) -> bool:
#     """对比两个tensor"""
#     print(f"\nTensor 1 shape: {tensor1.shape}, dtype: {tensor1.dtype}")
#     print(f"Tensor 2 shape: {tensor2.shape}, dtype: {tensor2.dtype}")
    
#     # 检查shape是否相同
#     if tensor1.shape != tensor2.shape:
#         print(f"❌ Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
#         return False
    
#     # 检查dtype是否相同
#     if tensor1.dtype != tensor2.dtype:
#         print(f"⚠️  Dtype mismatch: {tensor1.dtype} vs {tensor2.dtype}")
#         print("Converting to same dtype for comparison...")
#         # 转换为float32进行比较
#         tensor1 = tensor1.float()
#         tensor2 = tensor2.float()
    
#     # 完全相等检查
#     if torch.equal(tensor1, tensor2):
#         print("✅ Tensors are EXACTLY equal!")
#         return True
    
#     # 近似相等检查
#     if torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
#         print(f"✅ Tensors are approximately equal (rtol={rtol}, atol={atol})")
        
#         # 计算差异统计
#         diff = torch.abs(tensor1 - tensor2)
#         print(f"   Max absolute difference: {diff.max().item():.2e}")
#         print(f"   Mean absolute difference: {diff.mean().item():.2e}")
#         print(f"   Number of different elements: {(diff > atol).sum().item()}")
        
#         return True
    
#     # 不相等，显示详细差异
#     print("❌ Tensors are NOT equal!")
#     diff = torch.abs(tensor1 - tensor2)
#     max_diff = diff.max().item()
#     mean_diff = diff.mean().item()
#     num_diff = (diff > atol).sum().item()
#     total_elements = tensor1.numel()
    
#     print(f"   Max absolute difference: {max_diff:.2e}")
#     print(f"   Mean absolute difference: {mean_diff:.2e}")
#     print(f"   Number of different elements: {num_diff} / {total_elements} ({100*num_diff/total_elements:.2f}%)")
    
#     # 找出差异最大的位置
#     max_diff_idx = torch.unravel_index(diff.argmax(), diff.shape)
#     print(f"   Max diff location: {max_diff_idx}")
#     print(f"   Value at max diff location - tensor1: {tensor1[max_diff_idx].item():.6e}")
#     print(f"   Value at max diff location - tensor2: {tensor2[max_diff_idx].item():.6e}")
    
#     # 显示一些统计信息
#     print(f"\n   Tensor 1 stats:")
#     print(f"      Min: {tensor1.min().item():.6e}, Max: {tensor1.max().item():.6e}, Mean: {tensor1.mean().item():.6e}")
#     print(f"   Tensor 2 stats:")
#     print(f"      Min: {tensor2.min().item():.6e}, Max: {tensor2.max().item():.6e}, Mean: {tensor2.mean().item():.6e}")
    
#     return False


# def compare_dicts(dict1: dict, dict2: dict, rtol: float, atol: float) -> bool:
#     """对比两个dict"""
#     keys1 = set(dict1.keys())
#     keys2 = set(dict2.keys())
    
#     if keys1 != keys2:
#         print(f"❌ Key mismatch!")
#         print(f"   Keys only in file1: {keys1 - keys2}")
#         print(f"   Keys only in file2: {keys2 - keys1}")
#         return False
    
#     all_equal = True
#     for key in keys1:
#         print(f"\nComparing key: '{key}'")
#         if isinstance(dict1[key], torch.Tensor) and isinstance(dict2[key], torch.Tensor):
#             if not compare_tensors(dict1[key], dict2[key], rtol, atol):
#                 all_equal = False
#         elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
#             if not compare_dicts(dict1[key], dict2[key], rtol, atol):
#                 all_equal = False
#         else:
#             if dict1[key] != dict2[key]:
#                 print(f"❌ Values for key '{key}' are different")
#                 all_equal = False
    
#     return all_equal


# def compare_lists(list1, list2, rtol: float, atol: float) -> bool:
#     """对比两个list"""
#     if len(list1) != len(list2):
#         print(f"❌ Length mismatch: {len(list1)} vs {len(list2)}")
#         return False
    
#     all_equal = True
#     for i, (item1, item2) in enumerate(zip(list1, list2)):
#         print(f"\nComparing item {i}:")
#         if isinstance(item1, torch.Tensor) and isinstance(item2, torch.Tensor):
#             if not compare_tensors(item1, item2, rtol, atol):
#                 all_equal = False
#         elif isinstance(item1, dict) and isinstance(item2, dict):
#             if not compare_dicts(item1, item2, rtol, atol):
#                 all_equal = False
#         else:
#             if item1 != item2:
#                 print(f"❌ Items at index {i} are different")
#                 all_equal = False
    
#     return all_equal


# # 使用示例
# if __name__ == "__main__":
#     file1 = "/mnt/zhangshijin/vllm-omni-dev/codes_packed.pt"
#     file2 = "/mnt/zhangshijin/MiMo-Audio/codes_packed.pt"
    
#     print("=" * 80)
#     print("Comparing PyTorch tensor files")
#     print("=" * 80)
    
#     result = compare_pt_files(file1, file2)
    
#     print("\n" + "=" * 80)
#     if result:
#         print("✅ Files are EQUAL (or approximately equal)")
#     else:
#         print("❌ Files are DIFFERENT")
#     print("=" * 80)

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
from vllm_omni.model_executor.models.mimo_audio.mimo_audio_code2wav import MiMoAudioTokenizerWorker, get_tokenizer_worker
import torch
from torch.cuda import CUDAGraph

device = torch.device("cuda:0")
config_path = "/mnt/zhangshijin/vllm-omni-dev/models/MiMo-Audio-Tokenizer"
audio_tokenizer_path = "/mnt/zhangshijin/vllm-omni-dev/models/MiMo-Audio-Tokenizer"

tokenizer_service: MiMoAudioTokenizerWorker | None = get_tokenizer_worker(
            device=device,
            config_path=config_path,
            audio_tokenizer_path=audio_tokenizer_path,
        )

audio_tokenizer = tokenizer_service.audio_tokenizer
codes = torch.zeros(8,40,dtype=torch.long,device=device)
static_input_length = torch.full(
            (1,), 4, dtype=torch.long, device=device
        )

graph = CUDAGraph()
stream = torch.cuda.Stream(device=device)
with torch.cuda.graph(graph, stream=stream):
    static_output = audio_tokenizer.decode(codes, static_input_length=static_input_length)
print(static_output.shape)