#BY orange with Deepseek-R1 2025/4/4
#矿石X射线衍射谱可视化与音频转化
import os
import numpy as np
import matplotlib.pyplot as plt
from midiutil import MIDIFile
from scipy.signal import find_peaks, savgol_filter
import matplotlib.font_manager as fm

class Config:
    def __init__(self):
        # 文件参数
        self.input_dir = "XRD_Data"  # 输入数据目录
        self.output_dir = "XRay_Music"  # 输出数据目录
        
        # 音频参数
        self.max_duration = 90           # 最大时长(秒)
        self.base_bpm = 80               # 基础节奏
        self.note_duration = 2        # 基础音符时长
        self.dynamic_range = (60, 100)   # 音量动态范围
        
        # 音色配置 (GM标准)
        self.instruments = {
            "main": 87,      # 主旋律：科幻音效(FX1)
            "ambient": 103,  # 环境层：太空氛围(FX3)
            "bass": 24,      # 低频层：尼龙吉他
            "effect": 101    # 特效层：高频脉冲(FX4)
        }
        
        # 音高映射
        self.pitch_curve = lambda x: int(30 + 5*(x**0.5))  # 非线性音高函数
        
        # 可视化
        self.plot_style = "dark_background"  # 内置暗色主题
        self.font_path = "C://Windows//Fonts//msyh.ttc"         # 字体
        self.fallback_font = "SimHei"       # 备用字体

class XRaySonifier:
    def __init__(self, cfg):
        # 初始化方法，用于创建类的实例
        # 参数cfg：配置对象，包含类的配置信息
        self.cfg = cfg
        # 将传入的配置对象赋值给实例变量self.cfg，以便在类的其他方法中使用
        self._setup_directories()
        # 调用实例方法_setup_directories()，用于设置或创建必要的目录结构
        self._configure_visualization()
        
    def _setup_directories(self):
        """创建输出目录结构"""
        # 使用os.makedirs创建目录，os.path.join用于拼接路径
        # self.cfg.output_dir是配置中的输出目录路径
        # "Audio"和"Report"是子目录名
        # exist_ok=True表示如果目录已存在，不会抛出异常
        os.makedirs(os.path.join(self.cfg.output_dir, "Audio"), exist_ok=True)
        os.makedirs(os.path.join(self.cfg.output_dir, "Report"), exist_ok=True)
        
    def _configure_visualization(self):
        """配置可视化环境"""
        try:
            # 尝试使用配置文件中指定的绘图样式
            plt.style.use(self.cfg.plot_style)
        except:
            # 如果指定样式无法使用，则打印警告信息并使用默认样式
            print("⚠ 使用默认绘图样式")
        
        # 中文字体配置
        try:
            # 尝试使用配置文件中指定的字体文件路径
            font = fm.FontProperties(fname=self.cfg.font_path)
            # 设置matplotlib的字体属性为指定字体
            plt.rcParams['font.sans-serif'] = [font.get_name()]
            # 禁用负号使用Unicode编码，以支持中文显示
            plt.rcParams['axes.unicode_minus'] = False
        except:
            # 如果指定字体文件路径无效，尝试使用备用字体
            try:
                plt.rcParams['font.sans-serif'] = [self.cfg.fallback_font]
                print("⚠ 使用备用中文字体")
            except:
                # 如果备用字体也无效，则打印配置失败的信息
                print("⚠ 中文字体配置失败")

    def _load_data(self, file_path):
        """加载并验证数据文件"""
        try:
            # 使用numpy的loadtxt函数从指定文件路径加载数据，数据以逗号分隔
            data = np.loadtxt(file_path, delimiter=',')
            # 检查数据的维度是否为2（即二维数组）以及每行是否有2列
            if data.ndim != 2 or data.shape[1] != 2:
                # 如果数据格式不符合要求，抛出ValueError异常
                raise ValueError("数据格式错误，需要两列数据")
            # 返回数据的第一列（角度）和第二列（强度）
            return data[:,0], data[:,1]  # 角度, 强度
        except Exception as e:
            # 如果在加载或验证数据过程中发生任何异常，打印错误信息
            print(f"数据加载失败: {str(e)}")
            # 返回None, None表示数据加载失败
            return None, None
            
    def _extract_features(self, angles, intensities):
        """特征提取与评分"""
        # 数据平滑
        # 使用Savitzky-Golay滤波器对强度数据进行平滑处理，以减少噪声
        # window_length=11 表示滤波器窗口大小为11
        # polyorder=3 表示多项式的阶数为3
        smoothed = savgol_filter(intensities, window_length=11, polyorder=3)
        
        # 找极值点
        # 使用find_peaks函数找到平滑后数据的峰值点
        # prominence参数设置峰值的突出度，这里设置为平滑数据标准差的0.5倍
        peaks, _ = find_peaks(smoothed, prominence=np.std(smoothed)*0.5)
        # 使用find_peaks函数找到平滑后数据的谷值点
        # 通过对平滑数据取负值，再找峰值，从而实现找谷值
        valleys, _ = find_peaks(-smoothed, prominence=np.std(smoothed)*0.5)
        
        # 合并特征点
        # 初始化特征点列表
        features = []
        # 遍历峰值点和谷值点的索引集合
        for idx in set(peaks.tolist() + valleys.tolist()):
            # 计算角度与90度的偏差
            dev = abs(angles[idx] - 90)
            score = smoothed[idx] * (1 + dev/10)  # 评分公式
            features.append({
                "angle": angles[idx],
                "intensity": smoothed[idx],
                "deviation": dev,
                "score": score
            })
        return sorted(features, key=lambda x: -x["score"])  # 按评分排序
    
    def _generate_events(self, features):
        """增加参数校验"""
        # 初始化事件列表
        events = []
        # 生成时间点，从0.1秒开始，到最大持续时间，均匀分布
        time_points = np.linspace(0.1, self.cfg.max_duration, len(features))  # 从0.1秒开始
        
        for t, f in zip(time_points, features):
            # 校验音高范围
            # 根据特征中的偏差计算原始音高
            raw_pitch = self.cfg.pitch_curve(f["deviation"])
            # 将原始音高限制在0到127之间
            pitch = np.clip(raw_pitch, 0, 127)
            
            # 校验音量范围
            # 将特征中的强度值线性插值到配置的动态范围
            raw_vol = int(np.interp(
                f["intensity"],
                [min(f["intensity"] for f in features), 
                max(f["intensity"] for f in features)],
                self.cfg.dynamic_range
            ))
            # 将音量限制在0到127之间
            volume = np.clip(raw_vol, 0, 127)
            
            # 将事件添加到事件列表中
            events.append({
                "time": t,  # 事件发生的时间点
                "duration": np.clip(
                    self.cfg.note_duration * (0.5 + f["score"]/max(f["score"] for f in features)),
                    0.5,  # 最小持续时间
                    self.cfg.max_duration  # 最大不超过总时长
                ),
                "pitch": pitch,  # 音高
                "velocity": volume  # 修正参数名称
            })
        return events

    def _add_ambient_effects(self, midi, duration):
        midi.addControllerEvent(
            track=2,
            channel=0,
            time=0,
            controller_number=91,  # 混响效果器
            parameter=60          # 修复参数名称，值限制在0-127
        )
        
        # 优化粒子音效时间（避免0点开始）
        for _ in range(int(duration/3)):#/n n越小越多
            start_time = np.random.uniform(0.5, duration-2)  # 从1秒开始
            midi.addNote(
                track=3,
                channel=0,
                pitch=np.random.choice([84, 86, 89]),
                time=start_time,
                duration=0.15,
                volume=np.clip(np.random.randint(30, 50), 0, 127)  # 限制音量范围
            )
        
        # 低频震动优化
        for t in np.arange(0.2, duration, 4):  # 从0.2秒开始
            midi.addNote(
                track=2,
                channel=0,
                pitch=36,
                time=t,
                duration=3,
                volume=np.clip(35, 0, 127)
            )
    
    def create_midi(self, events, duration, filename):
        """生成完整MIDI文件"""
        # 创建一个MIDI文件对象，包含4个音轨
        midi = MIDIFile(numTracks=4)
        
        # 配置音色
        # 遍历配置文件中的乐器设置，为每个乐器设置音色和初始速度
        for i, (name, prog) in enumerate(self.cfg.instruments.items()):
            # 为第i个音轨设置音色，音色编号为prog
            midi.addProgramChange(i, 0, 0, prog)
            # 为第i个音轨设置初始速度，速度为self.cfg.base_bpm
            midi.addTempo(i, 0, self.cfg.base_bpm)
        
        # 主旋律
        # 遍历事件列表，为每个事件添加音符
        for evt in events:
            # 添加音符到第0个音轨，通道为0
            midi.addNote(
                track=0, channel=0,
                # 音符的音高
                pitch=evt["pitch"],
                # 音符的开始时间
                time=evt["time"],
                # 音符的持续时间
                duration=evt["duration"],
                # 音符的音量
                volume=evt["velocity"]
            )
        
        # 添加环境音效
        # 调用私有方法_add_ambient_effects，为MIDI文件添加环境音效
        self._add_ambient_effects(midi, duration)
        
        # 保存文件
        # 构建输出文件的路径
        output_path = os.path.join(self.cfg.output_dir, "Audio", f"{filename}.mid")
        # 以二进制写模式打开文件
        with open(output_path, "wb") as f:
            # 将MIDI文件写入到文件中
            midi.writeFile(f)
        # 返回输出文件的路径
        return output_path
    
    def generate_report(self, angles, intensities, features, filename):
        """生成分析报告"""
        # 创建一个新的图形，并设置图形的大小为16x9英寸
        plt.figure(figsize=(16, 9))
        
        # 主图：数据曲线
        # 创建一个2行1列的子图，当前绘制第一个子图
        plt.subplot(2,1,1)
        # 绘制角度和强度之间的关系曲线，颜色为青色，透明度为0.3，标签为“原始数据”
        plt.plot(angles, intensities, 'cyan', alpha=0.3, label='原始数据')
        # 绘制特征点的散点图，颜色根据特征点的偏差值映射，大小为50，边缘颜色为白色，标签为“特征点”
        plt.scatter(
            [f["angle"] for f in features],  # 提取特征点的角度
            [f["intensity"] for f in features],  # 提取特征点的强度
            c=[f["deviation"] for f in features],  # 提取特征点的偏差值作为颜色映射
            cmap='viridis',  # 使用viridis颜色映射
            s=50,  # 散点大小
            edgecolor='w',  # 散点边缘颜色为白色
            label='特征点'  # 标签为“特征点”
        )
        # 添加颜色条，标签为“角度偏差”
        plt.colorbar(label='角度偏差')
        # 绘制垂直线，位置为90度，颜色为红色，虚线样式，透明度为0.5
        plt.axvline(90, color='red', linestyle='--', alpha=0.5)
        # 设置子图的标题为“X射线衍射特征分析”
        plt.title("X射线衍射特征分析")
        # 添加图例
        plt.legend()
        
        # 音高分布
        # 创建一个2行1列的子图，当前绘制第二个子图
        plt.subplot(2,1,2)
        # 计算特征点的音高值
        pitches = [self.cfg.pitch_curve(f["deviation"]) for f in features]
        # 绘制音高的直方图，24个柱子，颜色为洋红色，透明度为0.7
        plt.hist(pitches, bins=24, color='magenta', alpha=0.7)
        # 设置X轴标签为“MIDI音高”
        plt.xlabel("MIDI音高")
        # 设置Y轴标签为“频次”
        plt.ylabel("频次")
        # 设置子图的标题为“音高分布”
        plt.title("音高分布")
        
        # 保存报告
        # 构建报告的保存路径
        report_path = os.path.join(self.cfg.output_dir, "Report", f"{filename}.png")
        # 调整子图布局，避免重叠
        plt.tight_layout()
        # 保存图形到指定路径，分辨率为100 DPI
        plt.savefig(report_path, dpi=100)
        # 关闭当前图形，释放内存
        plt.close()
        # 返回报告的保存路径
        return report_path
    def process_files(self):
        """批量处理输入目录中的所有文件"""
        # 遍历输入目录中的所有文件
        for filename in os.listdir(self.cfg.input_dir):
            # 检查文件是否为文本文件
            if filename.endswith('.txt'):
                file_path = os.path.join(self.cfg.input_dir, filename)
                # 处理单个文件
                self.process_file(file_path)
            else:
                print(f"跳过非文本文件: {filename}")

    def process_file(self, file_path):
        """完整处理流程"""
        # 1. 加载数据
        # 调用_load_data方法从文件中加载数据，返回角度和强度
        angles, intensities = self._load_data(file_path)
        # 如果加载的数据为空，则返回False，表示处理失败
        if angles is None:
            return False
        
        # 2. 特征提取
        # 调用_extract_features方法从角度和强度中提取特征
        features = self._extract_features(angles, intensities)
        # 如果提取的特征为空，则打印提示信息并返回False
        if not features:
            print("⚠ 未检测到有效特征")
            return False
        
        # 3. 生成音乐事件
        # 调用_generate_events方法从特征中生成音乐事件，只取前50个特征点
        events = self._generate_events(features[:50])  # 取前50个特征点
        
        # 4. 创建MIDI
        # 调用create_midi方法生成MIDI文件，传入音乐事件、最大持续时间和文件名
        midi_path = self.create_midi(events, self.cfg.max_duration, 
                                   os.path.splitext(os.path.basename(file_path))[0])
        
        # 5. 生成报告
        # 调用generate_report方法生成分析报告，传入角度、强度、特征和文件名
        report_path = self.generate_report(angles, intensities, features,
                                         os.path.splitext(os.path.basename(file_path))[0])
        
        # 打印生成成功的提示信息，包括MIDI文件路径和分析报告路径
        print(f"✅ 生成成功！\nMIDI文件: {midi_path}\n分析报告: {report_path}")
        return True

if __name__ == "__main__":
    print("====== X射线音乐化系统 ======")
    
    # 初始化配置
    config = Config()
    
    # 检查字体
    if not os.path.exists(config.font_path):
        print(f"⚠ 请下载微软雅黑字体并放置于: {os.path.abspath(config.font_path)}")
    
    try:
        # 创建处理器
        processor = XRaySonifier(config)
        
        # 批量处理输入目录中的所有文件
        processor.process_files()
        print("🎉 处理完成！")
            
    except Exception as e:
        print(f"❌ 系统错误: {str(e)}")