"""
电池SOH预测系统 - 启动脚本
"""

import subprocess
import sys
import os

def check_dependencies():
    """检查并安装依赖"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  缺少以下依赖包：{', '.join(missing_packages)}")
        print("📦 正在安装依赖包...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} 安装成功")
            except subprocess.CalledProcessError:
                print(f"❌ {package} 安装失败")
                print(f"💡 请手动运行: pip install {package}")
                return False
    
    return True

def start_app():
    """启动Streamlit应用"""
    if not check_dependencies():
        print("\n❌ 依赖安装失败，无法启动应用")
        return
    
    print("\n" + "="*60)
    print("🔋 电池健康状态（SOH）预测系统")
    print("="*60)
    print("\n📝 使用说明：")
    print("1. 应用将在浏览器中自动打开")
    print("2. 可以使用示例数据或上传自己的CSV文件")
    print("3. 支持多种机器学习模型进行预测")
    print("4. 包含完整的数据预处理和特征工程")
    print("\n💡 提示：按 Ctrl+C 停止应用")
    print("="*60 + "\n")
    
    try:
        # 启动Streamlit应用
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'battery_soh_prediction.py'])
    except KeyboardInterrupt:
        print("\n\n👋 应用已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("💡 您可以手动运行：streamlit run battery_soh_prediction.py")

if __name__ == '__main__':
    start_app()
