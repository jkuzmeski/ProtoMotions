"""Compatibility helpers for Newton's windowed OpenGL viewer."""

from __future__ import annotations

import os
import sys
from typing import Any


def prepare_gl_viewer_compat() -> None:
    """Apply compatibility shims required by Newton's GL viewer."""
    _disable_cuda_gl_interop_when_needed()
    _ensure_pyglet_current_context_fallback()
    _ensure_imgui_color_edit_compat()


def _ensure_imgui_color_edit_compat() -> None:
    """Accept RGB tuples in imgui_bundle's newer `color_edit3` binding."""
    try:
        from imgui_bundle import imgui
    except ImportError:
        return

    color_edit3 = getattr(imgui, "color_edit3", None)
    if color_edit3 is None:
        return

    if getattr(color_edit3, "_protomotions_tuple_compat", False):
        return

    def _color_edit3_with_tuple_compat(label, col, flags=0):
        if isinstance(col, imgui.ImVec4):
            return color_edit3(label, col, flags)

        channels = _coerce_color_channels(col)
        if channels is None:
            return color_edit3(label, col, flags)

        rgba = imgui.ImVec4(
            float(channels[0]),
            float(channels[1]),
            float(channels[2]),
            float(channels[3]) if len(channels) == 4 else 1.0,
        )
        changed, value = color_edit3(label, rgba, flags)
        return changed, (value.x, value.y, value.z)

    _color_edit3_with_tuple_compat._protomotions_tuple_compat = True
    imgui.color_edit3 = _color_edit3_with_tuple_compat


def _coerce_color_channels(value: Any) -> tuple[Any, ...] | None:
    """Return RGB(A) channels when `value` looks like a color sequence."""
    if isinstance(value, (str, bytes)):
        return None

    try:
        channels = tuple(value)
    except TypeError:
        return None

    if len(channels) not in (3, 4):
        return None

    return channels


def _ensure_pyglet_current_context_fallback() -> None:
    """Let PyOpenGL fall back to Pyglet's active window context.

    On some Linux/WSL setups, PyOpenGL picks an EGL platform backend even when
    Newton's `ViewerGL` opens a windowed Pyglet/GLX context. In that case
    `OpenGL.platform.GetCurrentContext()` returns `None`, which breaks ImGui's
    OpenGL backend during viewer startup. This wrapper preserves the original
    lookup and only falls back to `pyglet.gl.current_context` when needed.
    """
    try:
        import OpenGL.platform as opengl_platform
        import pyglet.gl as pyglet_gl
    except ImportError:
        return

    get_current_context = getattr(opengl_platform, "GetCurrentContext", None)
    if get_current_context is None:
        return

    if getattr(get_current_context, "_protomotions_pyglet_fallback", False):
        return

    def _get_current_context_with_pyglet_fallback():
        context = get_current_context()
        if context:
            return context
        return getattr(pyglet_gl, "current_context", None)

    _get_current_context_with_pyglet_fallback._protomotions_pyglet_fallback = True

    opengl_platform.GetCurrentContext = _get_current_context_with_pyglet_fallback

    platform_impl = getattr(opengl_platform, "PLATFORM", None)
    if platform_impl is not None:
        platform_impl.GetCurrentContext = _get_current_context_with_pyglet_fallback


def _disable_cuda_gl_interop_when_needed() -> None:
    """Disable Newton's CUDA-GL interop path on setups where it is noisy or unsupported."""
    if not _should_disable_cuda_gl_interop():
        return

    newton_opengl = sys.modules.get("newton._src.viewer.gl.opengl")
    if newton_opengl is None:
        try:
            from newton._src.viewer.gl import opengl as newton_opengl
        except ImportError:
            return

    newton_opengl.ENABLE_CUDA_INTEROP = False


def _should_disable_cuda_gl_interop() -> bool:
    """Return whether Newton's CUDA-GL interop path should be disabled."""
    disable_override = os.getenv("PROTOMOTIONS_DISABLE_NEWTON_CUDA_GL_INTEROP")
    if disable_override is not None:
        return disable_override.lower() not in {"0", "false", "no"}

    if os.getenv("WSL_INTEROP") or os.getenv("WSL_DISTRO_NAME"):
        return True

    return False
