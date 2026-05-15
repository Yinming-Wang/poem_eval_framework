"""theme_generation 任务的数据构造模块。"""

from __future__ import annotations

from typing import Dict, List


COMMON_THEMES = [
    # 四季节令
    "春天", "夏日", "秋思", "冬雪", "春雨", "秋风", "寒冬", "新年", "清明", "中秋",

    # 自然景物
    "明月", "清风", "山水", "江河", "大海", "白云", "落日", "晚霞", "青山", "流水",
    "花开", "落花", "杨柳", "孤舟", "飞鸟", "寒梅", "竹林", "松柏", "荷花", "枫叶",

    # 情感主题
    "思乡", "离别", "相思", "孤独", "怀古", "忧愁", "喜悦", "闲适", "感怀", "伤春",

    # 人生与哲理
    "人生", "时光", "梦想", "远方", "归隐", "漂泊", "壮志", "知己", "故人", "岁月",

    # 场景主题
    "边塞", "田园", "江南", "长安", "古寺", "夜雨", "客栈", "渡口", "山村", "渔家",

    # 文化意象
    "美酒", "琴声", "书卷", "灯火", "诗酒", "征人", "游子", "故乡", "关山", "楼台",

    # 扩展主题
    "晨景", "暮景", "湖上", "溪边", "山寺", "秋夜", "春游", "送别", "登高", "望月",
    "听雨", "赏花", "观潮", "塞外", "军旅", "宫怨", "闺思", "农家", "渔舟", "山居",
    "旅夜", "夏雨", "冬夜", "梅雪", "兰亭", "边月", "江雪", "云梦", "雨巷", "古道",
]


def build_theme_generation_samples(num_samples: int = 1000) -> List[Dict]:
    """
    根据 COMMON_THEMES 自动构造 theme_generation 测试样本。

    默认构造 1000 条样本。主题词会循环复用，诗体按样本序号交替分配为
    “五言绝句”或“七言绝句”。后续关键词生成、上句续写和风格控制任务
    可以复用这些保留字段。
    """
    if num_samples <= 0:
        num_samples = len(COMMON_THEMES)

    samples: List[Dict] = []
    for index in range(num_samples):
        theme = COMMON_THEMES[index % len(COMMON_THEMES)]
        poem_type = "五言绝句" if index % 2 == 0 else "七言绝句"
        samples.append(
            {
                "id": "T{0:04d}".format(index + 1),
                "task_type": "theme_generation",
                "theme": theme,
                "keywords": [],
                "prefix": "",
                "style": "",
                "poem_type": poem_type,
            }
        )
    return samples
