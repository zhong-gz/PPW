import subprocess
import sys

# 在这里填写你要运行的Python文件路径！
files_to_run = [
    # "SyntheticDatasets/linear_regression.py",        
    "Communities and Crime/CommunitiesAndCrimeSimulation.py",  
    "Communities and Crime/par_sensetivie.py",                 
    "StudentMarks/StudentMarksSimulation.py",
    "StoreDemand/StoreDemand.py",
]

if __name__ == "__main__":
    python_executable = sys.executable  # 使用当前Python解释器
    
    for file_path in files_to_run:
        try:
            # 执行文件（忽略输出）
            subprocess.run(
                [python_executable, file_path],
                # stdout=subprocess.DEVNULL,
                # stderr=subprocess.DEVNULL,
                # check=True
            )
            print(f"成功执行: {file_path}")
        except Exception as e:
            print(f"执行失败: {file_path} ({str(e)})")