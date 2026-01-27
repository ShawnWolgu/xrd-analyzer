# convert_csv_to_txt.py - CSV转TXT格式转换器

"""
将CSV格式的XRD数据转换为标准TXT格式

输入格式: angle, intensity (每行)
输出格式: angle\tintensity (制表符分隔)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def convert_csv_to_txt(csv_path, output_path=None, sort_by_angle=True):
    """
    转换CSV到TXT格式
    
    Parameters:
    -----------
    csv_path : str
        输入CSV文件路径
    output_path : str, optional
        输出TXT文件路径，如果为None则自动生成
    sort_by_angle : bool
        是否按角度排序（推荐True）
    """
    
    print(f"读取文件: {csv_path}")
    
    # 读取CSV（无表头）
    try:
        data = pd.read_csv(csv_path, header=None, names=['angle', 'intensity'])
    except Exception as e:
        print(f"读取失败: {e}")
        return
    
    print(f"  读取了 {len(data)} 行数据")
    
    # 检查数据
    if data.isnull().any().any():
        print("  警告: 数据中包含空值，将被移除")
        data = data.dropna()
    
    # 按角度排序
    if sort_by_angle:
        data = data.sort_values('angle')
        print("  数据已按角度排序")
    
    # 去除重复的角度（如果有）
    if data['angle'].duplicated().any():
        print(f"  警告: 发现 {data['angle'].duplicated().sum()} 个重复角度")
        print("  将对重复角度的强度取平均值")
        data = data.groupby('angle', as_index=False).mean()
    
    # 生成输出路径
    if output_path is None:
        input_path = Path(csv_path)
        output_path = input_path.parent / f"{input_path.stem}_converted.txt"
    
    # 写入TXT文件（制表符分隔）
    with open(output_path, 'w') as f:
        for _, row in data.iterrows():
            f.write(f"{row['angle']:.6f}\t{row['intensity']:.6f}\n")
    
    print(f"\n转换完成！")
    print(f"输出文件: {output_path}")
    print(f"数据范围: {data['angle'].min():.2f}° - {data['angle'].max():.2f}°")
    print(f"数据点数: {len(data)}")
    
    return output_path


def batch_convert(folder_path, pattern="*.csv"):
    """
    批量转换文件夹中的所有CSV文件
    
    Parameters:
    -----------
    folder_path : str
        文件夹路径
    pattern : str
        文件匹配模式
    """
    folder = Path(folder_path)
    csv_files = list(folder.glob(pattern))
    
    if not csv_files:
        print(f"在 {folder_path} 中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件\n")
    
    for csv_file in csv_files:
        print("="*60)
        convert_csv_to_txt(str(csv_file))
        print()
    
    print("="*60)
    print("批量转换完成！")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 命令行模式
        input_file = sys.argv[1]
        
        if Path(input_file).is_dir():
            # 批量转换
            batch_convert(input_file)
        else:
            # 单文件转换
            output_file = sys.argv[2] if len(sys.argv) > 2 else None
            convert_csv_to_txt(input_file, output_file)
    else:
        # 交互模式
        print("="*60)
        print("CSV到TXT格式转换器")
        print("="*60)
        
        input_path = input("\n请输入CSV文件路径（或文件夹路径进行批量转换）: ").strip().strip('"').strip("'")
        
        if not Path(input_path).exists():
            print(f"错误: 路径不存在 - {input_path}")
            sys.exit(1)
        
        if Path(input_path).is_dir():
            batch_convert(input_path)
        else:
            convert_csv_to_txt(input_path)

