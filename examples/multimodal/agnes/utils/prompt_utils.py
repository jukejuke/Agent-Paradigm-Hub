"""
Agnes 提示词构建工具
====================

提供风格迁移与文字增 / 改 / 删三类提示词的统一拼装模板，便于复用与调参。

写词要点：
- 风格提示词：目标风格 + 保持主体 / 构图 + 画质词，效果更稳定
- 文字提示词：把要渲染 / 替换的文字内容显式写出（用「」标注），
  并强调「文字必须完全一致、无错别字」，可显著提升渲染准确率

作者: Sol
日期: 2026-09-16
"""

# 文字渲染强化语：显式要求文字完全一致、无错别字
_TEXT_ACCURACY_HINT = "文字必须完全一致、无错别字"


def build_style_prompt(style_desc: str, keep_hint: str = "保留主体与构图") -> str:
    """
    构建风格迁移提示词

    Args:
        style_desc: 目标风格描述，如 "中国传统水墨画，宣纸质感，浓淡墨色晕染"
        keep_hint: 保持画面一致性的约束语，默认 "保留主体与构图"

    Returns:
        风格迁移提示词字符串
    """
    return f"将参考图转换为{style_desc}风格，{keep_hint}，超高清"


def build_text_add_prompt(text: str, scene_hint: str = "") -> str:
    """
    构建『在图片中添加文字』提示词（用于文生图带文字）

    Args:
        text: 要渲染的文字内容
        scene_hint: 场景补充描述（背景 / 版式 / 风格等），可空

    Returns:
        加字提示词字符串
    """
    scene = f"，{scene_hint}" if scene_hint else ""
    return f"在图片中添加文字「{text}」{scene}，{_TEXT_ACCURACY_HINT}"


def build_text_replace_prompt(old_text: str, new_text: str) -> str:
    """
    构建『把图中文字 A 替换为 B』提示词（用于图生图改字）

    Args:
        old_text: 原文字内容
        new_text: 新文字内容

    Returns:
        改字提示词字符串
    """
    return (
        f"将图片中的文字「{old_text}」替换为「{new_text}」，"
        f"保持主体、版式与画风完全一致，{_TEXT_ACCURACY_HINT}"
    )


def build_text_remove_prompt(text: str) -> str:
    """
    构建『删除图中文字』提示词（用于图生图删字）

    Args:
        text: 要删除的文字内容

    Returns:
        删字提示词字符串
    """
    return (
        f"删除图片中的文字「{text}」，"
        f"用自然的背景 / 画面内容填补原文字区域，不留文字痕迹，保持主体与画风一致"
    )
