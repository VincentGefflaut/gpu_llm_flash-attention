
import torch
import pandas as pd
import os
import numpy as np
from softmax_matmul.softmax_matmul import fused_softmax, softmax_mult

try:
	from triton.compiler.errors import CompileTimeAssertionFailure
	from triton.runtime.errors import OutOfResources
except ImportError:
	CompileTimeAssertionFailure = type('CompileTimeAssertionFailure', (Exception,), {})
	OutOfResources = type('OutOfResources', (Exception,), {})

def measure_peak_memory():
	torch.cuda.synchronize()
	return torch.cuda.max_memory_allocated() / 1024 / 1024  # MiB

def benchmark_fn(fn, x, V, nb_warmup, nb_passes, BLOCK=None):
	torch.cuda.reset_peak_memory_stats()
	# Warmup
	for _ in range(nb_warmup):
		if BLOCK is not None:
			out = fn(x, V, BLOCK, BLOCK)
		else:
			out = fn(x, V)
	torch.cuda.synchronize()
	times = []
	for _ in range(nb_passes):
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)
		torch.cuda.synchronize()
		start.record()
		if BLOCK is not None:
			out = fn(x, V, BLOCK, BLOCK)
		else:
			out = fn(x, V)
		end.record()
		torch.cuda.synchronize()
		elapsed = start.elapsed_time(end)
		times.append(elapsed)
	peak_mem = measure_peak_memory()
	return np.mean(times), np.std(times), peak_mem

def main():
	device = "cuda"
	dtype = torch.float32
	batch_size = 16
	nb_warmup = 10
	nb_passes = 100
	d1 = 2048
	d2_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
	d3 = 512
	B_list = [16, 32, 64]

	results = []
	os.makedirs("outputs", exist_ok=True)

	# Triton (fused)
	for d2 in d2_list:
		for B in B_list:
			# Check divisibility
			if d1 % B != 0 or d2 % B != 0:
				print(f"Skipping: d1={d1}, d2={d2}, BLOCK={B} (not divisible)")
				results.append({
					"batch_size": batch_size,
					"d1": d1,
					"d2": d2,
					"d3": d3,
					"triton": True,
					"BLOCK": B,
					"forward_ms_mean": None,
					"forward_ms_std": None,
					"forward_peak_MiB": None,
				})
				continue
			x = torch.randn((batch_size, d1, d2), device=device, dtype=dtype)
			V = torch.randn((batch_size, d2, d3), device=device, dtype=dtype)
			try:
				mean, std, peak = benchmark_fn(fused_softmax, x, V, nb_warmup, nb_passes, BLOCK=B)
				results.append({
					"batch_size": batch_size,
					"d1": d1,
					"d2": d2,
					"d3": d3,
					"triton": True,
					"BLOCK": B,
					"forward_ms_mean": mean,
					"forward_ms_std": std,
					"forward_peak_MiB": peak,
				})
				print(f"[Triton] d2={d2}, BLOCK={B}: {mean:.2f}±{std:.2f} ms, {peak:.1f} MiB")
			except (RuntimeError, OutOfResources) as e:
				err_str = str(e).lower()
				if "out of memory" in err_str:
					print(f"OOM for d2={d2}, BLOCK={B}")
					torch.cuda.empty_cache()
					results.append({
						"batch_size": batch_size,
						"d1": d1,
						"d2": d2,
						"d3": d3,
						"triton": True,
						"BLOCK": B,
						"forward_ms_mean": None,
						"forward_ms_std": None,
						"forward_peak_MiB": None,
					})
				else:
					print(f"Error for d2={d2}, BLOCK={B}: {e}")
					results.append({
						"batch_size": batch_size,
						"d1": d1,
						"d2": d2,
						"d3": d3,
						"triton": True,
						"BLOCK": B,
						"forward_ms_mean": None,
						"forward_ms_std": None,
						"forward_peak_MiB": None,
					})
			except CompileTimeAssertionFailure:
				print(f"Skipping: BLOCK={B} incompatible with d2={d2}")
				results.append({
					"batch_size": batch_size,
					"d1": d1,
					"d2": d2,
					"d3": d3,
					"triton": True,
					"BLOCK": B,
					"forward_ms_mean": None,
					"forward_ms_std": None,
					"forward_peak_MiB": None,
				})

	# PyTorch (reference)
	for d2 in d2_list:
		x = torch.randn((batch_size, d1, d2), device=device, dtype=dtype)
		V = torch.randn((batch_size, d2, d3), device=device, dtype=dtype)
		try:
			mean, std, peak = benchmark_fn(softmax_mult, x, V, nb_warmup, nb_passes, BLOCK=None)
			results.append({
				"batch_size": batch_size,
				"d1": d1,
				"d2": d2,
				"d3": d3,
				"triton": False,
				"BLOCK": None,
				"forward_ms_mean": mean,
				"forward_ms_std": std,
				"forward_peak_MiB": peak,
			})
			print(f"[PyTorch] d2={d2}: {mean:.2f}±{std:.2f} ms, {peak:.1f} MiB")
		except (RuntimeError, OutOfResources) as e:
			err_str = str(e).lower()
			if "out of memory" in err_str:
				print(f"OOM for d2={d2} (PyTorch)")
				torch.cuda.empty_cache()
				results.append({
					"batch_size": batch_size,
					"d1": d1,
					"d2": d2,
					"d3": d3,
					"triton": False,
					"BLOCK": None,
					"forward_ms_mean": None,
					"forward_ms_std": None,
					"forward_peak_MiB": None,
				})
			else:
				print(f"Error for d2={d2} (PyTorch): {e}")
				results.append({
					"batch_size": batch_size,
					"d1": d1,
					"d2": d2,
					"d3": d3,
					"triton": False,
					"BLOCK": None,
					"forward_ms_mean": None,
					"forward_ms_std": None,
					"forward_peak_MiB": None,
				})

	df = pd.DataFrame(results)
	df["BLOCK"] = df["BLOCK"].astype("Int64")
	print(df.to_string())
	csv_path = os.path.join("outputs", "softmax_matmul_benchmark.csv")
	df.to_csv(csv_path, index=False)
	print(f"Results saved to {csv_path}")

if __name__ == "__main__":
	main()
