import mujoco
import sys

# 打印MuJoCo版本
print(f"MuJoCo版本: {mujoco.__version__}")

# 打印Python版本
print(f"Python版本: {sys.version}")

# 打印更详细的MuJoCo信息（如果可用）
if hasattr(mujoco, 'mj_versionString'):
    print(f"MuJoCo版本字符串: {mujoco.mj_versionString()}")

# 打印已安装的包信息
try:
    import pkg_resources
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    print("\n已安装的相关包:")
    for pkg in ['mujoco', 'mujoco-py', 'dm_control']:
        if pkg in installed_packages:
            print(f"  {pkg}: {installed_packages[pkg]}")
except ImportError:
    print("无法导入pkg_resources来检查已安装的包") 