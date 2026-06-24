# ==========================================
# ComfyUI-ComicOutline
# 漫画轮廓检测：多通道边缘融合 + 连通域去噪
# 纯 OpenCV 实现，无模型依赖
# ==========================================

_import_errors = []

try:
    from .nodes.ComicOutlineDetect import ComicOutlineDetect
except ImportError as e:
    _import_errors.append(f"ComicOutlineDetect: {e}")
    ComicOutlineDetect = None

if _import_errors:
    print(f"\033[33m[ComicOutline] 以下模块加载失败:\033[0m")
    for err in _import_errors:
        print(f"\033[33m  - {err}\033[0m")

# ==========================================
# 节点注册
# ==========================================
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if ComicOutlineDetect is not None:
    NODE_CLASS_MAPPINGS["ComicOutlineDetect"] = ComicOutlineDetect
    NODE_DISPLAY_NAME_MAPPINGS["ComicOutlineDetect"] = "🎨 漫画轮廓检测"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
