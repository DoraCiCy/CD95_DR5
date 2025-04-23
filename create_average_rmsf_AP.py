import numpy as np
import mdtraj as md

# 定义文件路径
repeat1_rmsf_file = "analysis_results_AP/repeat1_rmsf.xvg"
repeat2_rmsf_file = "analysis_results_AP/repeat2_rmsf.xvg"
repeat3_rmsf_file = "analysis_results_AP/repeat3_rmsf.xvg"
template_pdb_file = "analysis_results_AP/repeat1_rmsf.pdb"
output_pdb_file = "analysis_results_AP/average_rmsf.pdb"

# 加载RMSF数据
rmsf1 = np.loadtxt(repeat1_rmsf_file, comments=["@", "#"])
rmsf2 = np.loadtxt(repeat2_rmsf_file, comments=["@", "#"])
rmsf3 = np.loadtxt(repeat3_rmsf_file, comments=["@", "#"])

# 获取RMSF值
rmsf_val1 = rmsf1[:, 1]
rmsf_val2 = rmsf2[:, 1]
rmsf_val3 = rmsf3[:, 1]

# 确保所有数据长度一致
min_len = min(len(rmsf_val1), len(rmsf_val2), len(rmsf_val3))
rmsf_val1 = rmsf_val1[:min_len]
rmsf_val2 = rmsf_val2[:min_len]
rmsf_val3 = rmsf_val3[:min_len]

# 计算平均RMSF
avg_rmsf = np.mean([rmsf_val1, rmsf_val2, rmsf_val3], axis=0)

# 计算标准差（可用于误差分析）
std_rmsf = np.std([rmsf_val1, rmsf_val2, rmsf_val3], axis=0)

# 保存平均RMSF数据到文件
with open("analysis_results_AP/average_rmsf.dat", "w") as f:
    f.write("# Residue Average_RMSF StdDev_RMSF\n")
    for i, (avg, std) in enumerate(zip(avg_rmsf, std_rmsf)):
        f.write(f"{i+1} {avg:.5f} {std:.5f}\n")

# 加载模板PDB结构
structure = md.load(template_pdb_file)

# 为结构设置平均RMSF作为B-factor
# 先创建一个原子级别的B-factor数组
b_factors = np.zeros(structure.n_atoms)

# 为每个残基的所有原子设置相同的B-factor值
current_res = -1
res_index = -1

for i, atom in enumerate(structure.topology.atoms):
    res_id = atom.residue.index
    if res_id != current_res:
        current_res = res_id
        res_index += 1
        
    if res_index < len(avg_rmsf):
        b_factors[i] = avg_rmsf[res_index]

# 保存带有平均RMSF值作为B-factor的PDB文件
structure.save_pdb(output_pdb_file, bfactors=b_factors)
print(f"已成功创建平均RMSF的PDB文件：{output_pdb_file}")