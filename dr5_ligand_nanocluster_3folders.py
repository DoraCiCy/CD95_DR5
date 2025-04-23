"""
DR5激动剂纳米簇评估PyMOL脚本 - 限制分析"3"开头的文件夹
评估除ABC链外的剩余链(激动剂)是否形成规整对称的纳米簇结构
"""

import os
import glob
import numpy as np
import pymol
from pymol import cmd
import csv

def get_ligand_chains(object_name):
    """获取除ABC外的所有链(激动剂链)"""
    model = cmd.get_model(object_name)
    all_chains = set()
    
    for atom in model.atom:
        all_chains.add(atom.chain)
    
    # 排除ABC链，剩余链视为激动剂
    receptor_chains = {'A', 'B', 'C'}
    ligand_chains = all_chains - receptor_chains
    
    return sorted(list(ligand_chains))

def get_chain_centers(object_name, chain_ids):
    """获取指定链的几何中心"""
    centers = {}
    
    for chain in chain_ids:
        selection = f"{object_name} and chain {chain} and name CA"
        model = cmd.get_model(selection)
        coords = [atom.coord for atom in model.atom]
        if coords:
            centers[chain] = np.mean(coords, axis=0)
    
    return centers

def calculate_cluster_metrics(centers):
    """计算簇的对称性和规整性指标"""
    if len(centers) < 3:
        return {
            "symmetry": 0,
            "regularity": 0,
            "planarity": 0,
            "radial_distribution": 0
        }
    
    points = np.array(list(centers.values()))
    global_center = np.mean(points, axis=0)
    
    # 对称性 - 到中心的距离一致性
    distances = np.linalg.norm(points - global_center, axis=1)
    symmetry = 1.0 - (np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0)
    
    # 规整性 - 链间距离分布的一致性
    chain_distances = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            chain_distances.append(np.linalg.norm(points[i] - points[j]))
            
    regularity = 1.0 - (np.std(chain_distances) / np.mean(chain_distances) if np.mean(chain_distances) > 0 else 0)
    
    # 平面性 - 主分量分析
    centered_points = points - global_center
    cov = np.cov(centered_points.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    
    # 最小特征值占比越小，结构越平面
    planarity = 1.0 - (eig_vals[0] / sum(eig_vals) if sum(eig_vals) > 0 else 0)
    
    # 径向分布 - 角度分布的均匀性
    if len(points) > 3:
        vectors = points - global_center
        normalized_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        
        # 计算向量间夹角
        angles = []
        for i in range(len(normalized_vectors)):
            for j in range(i+1, len(normalized_vectors)):
                dot_product = np.dot(normalized_vectors[i], normalized_vectors[j])
                # 处理数值误差
                dot_product = max(-1.0, min(1.0, dot_product))
                angle = np.arccos(dot_product)
                angles.append(angle)
        
        radial_distribution = 1.0 - (np.std(angles) / np.mean(angles) if np.mean(angles) > 0 else 0)
    else:
        radial_distribution = 0
    
    return {
        "symmetry": symmetry,
        "regularity": regularity,
        "planarity": planarity,
        "radial_distribution": radial_distribution
    }

def analyze_ligand_clusters_3folders(folder_path):
    """分析以3开头的文件夹中DR5激动剂的簇结构特性"""
    # 获取所有以3开头的子文件夹
    all_subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    folders_3 = [f for f in all_subfolders if f.startswith('3')]
    
    if not folders_3:
        print(f"未找到以3开头的子文件夹")
        return
    
    print(f"找到{len(folders_3)}个以3开头的子文件夹")
    
    # 收集所有符合条件的CIF文件
    cif_files = []
    for subfolder in folders_3:
        subfolder_path = os.path.join(folder_path, subfolder)
        files = glob.glob(os.path.join(subfolder_path, "*_model_0.cif"))
        cif_files.extend(files)
    
    if not cif_files:
        print(f"在以3开头的子文件夹中未找到CIF文件")
        return
    
    print(f"找到{len(cif_files)}个CIF文件")
    results = []
    
    for cif_file in cif_files:
        folder = os.path.basename(os.path.dirname(cif_file))
        filename = os.path.basename(cif_file)
        
        print(f"分析: {folder}/{filename}")
        
        # 清理当前对象
        cmd.delete("all")
        
        try:
            # 加载结构
            cmd.load(cif_file, "structure")
            
            # 获取激动剂链
            ligand_chains = get_ligand_chains("structure")
            print(f"  发现{len(ligand_chains)}条激动剂链: {', '.join(ligand_chains)}")
            
            if len(ligand_chains) < 3:
                print("  警告: 激动剂链数量不足，无法形成有意义的簇")
                
                # 仍记录基本信息
                result = {
                    "folder": folder,
                    "file": filename,
                    "ligand_chain_count": len(ligand_chains),
                    "ligand_chains": ''.join(ligand_chains),
                    "symmetry": 0,
                    "regularity": 0,
                    "planarity": 0,
                    "radial_distribution": 0,
                    "nano_cluster_score": 0
                }
                results.append(result)
                continue
            
            # 获取激动剂链中心
            centers = get_chain_centers("structure", ligand_chains)
            
            # 计算簇特征
            metrics = calculate_cluster_metrics(centers)
            
            # 计算纳米簇评分 (综合多个指标)
            nano_cluster_score = (
                0.35 * metrics["symmetry"] + 
                0.35 * metrics["regularity"] + 
                0.15 * metrics["planarity"] + 
                0.15 * metrics["radial_distribution"]
            )
            
            # 记录结果
            result = {
                "folder": folder,
                "file": filename,
                "ligand_chain_count": len(ligand_chains),
                "ligand_chains": ''.join(ligand_chains),
                "symmetry": metrics["symmetry"],
                "regularity": metrics["regularity"],
                "planarity": metrics["planarity"],
                "radial_distribution": metrics["radial_distribution"],
                "nano_cluster_score": nano_cluster_score
            }
            
            results.append(result)
            print(f"  纳米簇评分: {nano_cluster_score:.3f}")
            print(f"  对称性: {metrics['symmetry']:.3f}, 规整性: {metrics['regularity']:.3f}")
            print(f"  平面性: {metrics['planarity']:.3f}, 径向分布: {metrics['radial_distribution']:.3f}")
        
        except Exception as e:
            print(f"  处理文件时出错: {str(e)}")
            continue
    
    if not results:
        print("没有成功分析任何文件")
        return
    
    # 保存结果
    output_csv = os.path.join(os.path.dirname(folder_path), "dr5_ligand_nanocluster_3folders.csv")
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            "folder", "file", "ligand_chain_count", "ligand_chains",
            "symmetry", "regularity", "planarity", "radial_distribution", 
            "nano_cluster_score"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"结果已保存至: {output_csv}")
    
    # 显示排名
    print("\n激动剂纳米簇评分排名前5的结构:")
    sorted_results = sorted(results, key=lambda x: x["nano_cluster_score"], reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        print(f"{i+1}. {r['folder']} - 评分: {r['nano_cluster_score']:.3f}, " + 
              f"激动剂链数: {r['ligand_chain_count']}, " +
              f"对称性: {r['symmetry']:.3f}, 规整性: {r['regularity']:.3f}")
    
    # 分析已知活性配体
    print("\n已知活性配体的纳米簇特征:")
    active_ligands = ["yyiysqtyfrfq", "erm_10", "rip_10", "tfreedspemcrk", "tyrfrq"]
    
    for ligand in active_ligands:
        matched_results = [r for r in results if ligand.lower() in r['folder'].lower()]
        if matched_results:
            for r in matched_results:
                print(f"{r['folder']} - 评分: {r['nano_cluster_score']:.3f}, " +
                     f"激动剂链数: {r['ligand_chain_count']}, " +
                     f"对称性: {r['symmetry']:.3f}")

print("DR5激动剂纳米簇评估脚本已加载 - 限制分析以3开头的文件夹")
print("使用方法: analyze_ligand_clusters_3folders('folds_2025_04_04_16_16')")