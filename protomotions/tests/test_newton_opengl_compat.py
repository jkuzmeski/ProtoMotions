import sys
from types import SimpleNamespace

from protomotions.simulator.newton.opengl_compat import (
    prepare_gl_viewer_compat,
)


def _install_fake_pyglet_context(monkeypatch, original_get_current_context, fallback_context):
    platform_impl = SimpleNamespace(GetCurrentContext=original_get_current_context)
    opengl_platform = SimpleNamespace(
        GetCurrentContext=original_get_current_context,
        PLATFORM=platform_impl,
    )
    pyglet_gl = SimpleNamespace(current_context=fallback_context)

    monkeypatch.setitem(sys.modules, "OpenGL.platform", opengl_platform)
    monkeypatch.setitem(sys.modules, "pyglet.gl", pyglet_gl)
    return opengl_platform


def test_prepare_gl_viewer_compat_uses_pyglet_when_needed(monkeypatch):
    fallback_context = object()

    original_calls = {"count": 0}

    def original_get_current_context():
        original_calls["count"] += 1
        return None

    opengl_platform = _install_fake_pyglet_context(
        monkeypatch, original_get_current_context, fallback_context
    )

    prepare_gl_viewer_compat()

    assert opengl_platform.GetCurrentContext() is fallback_context
    assert opengl_platform.PLATFORM.GetCurrentContext() is fallback_context
    assert original_calls["count"] == 2


def test_prepare_gl_viewer_compat_keeps_real_context(monkeypatch):
    existing_context = object()

    def original_get_current_context():
        return existing_context

    opengl_platform = _install_fake_pyglet_context(
        monkeypatch, original_get_current_context, object()
    )

    prepare_gl_viewer_compat()

    assert opengl_platform.GetCurrentContext() is existing_context
    assert opengl_platform.PLATFORM.GetCurrentContext() is existing_context


def test_prepare_gl_viewer_compat_wraps_rgb_tuples(monkeypatch):
    class FakeImVec4:
        def __init__(self, x, y, z, w):
            self.x = x
            self.y = y
            self.z = z
            self.w = w

    recorded = {}

    def original_color_edit3(label, col, flags=0):
        recorded["label"] = label
        recorded["col"] = col
        recorded["flags"] = flags
        return True, FakeImVec4(0.25, 0.5, 0.75, 1.0)

    _install_fake_pyglet_context(monkeypatch, lambda: None, object())
    imgui = SimpleNamespace(ImVec4=FakeImVec4, color_edit3=original_color_edit3)
    monkeypatch.setitem(sys.modules, "imgui_bundle", SimpleNamespace(imgui=imgui))

    prepare_gl_viewer_compat()

    changed, color = imgui.color_edit3("Light Color", (1.0, 1.0, 1.0), 7)

    assert changed is True
    assert color == (0.25, 0.5, 0.75)
    assert isinstance(recorded["col"], FakeImVec4)
    assert recorded["flags"] == 7


def test_prepare_gl_viewer_compat_is_idempotent(monkeypatch):
    def original_get_current_context():
        return None

    opengl_platform = _install_fake_pyglet_context(monkeypatch, original_get_current_context, object())
    imgui = SimpleNamespace(ImVec4=type("FakeImVec4", (), {}), color_edit3=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "imgui_bundle", SimpleNamespace(imgui=imgui))

    prepare_gl_viewer_compat()
    first_get_current_context = opengl_platform.GetCurrentContext
    first_color_edit3 = imgui.color_edit3

    prepare_gl_viewer_compat()

    assert opengl_platform.GetCurrentContext is first_get_current_context
    assert imgui.color_edit3 is first_color_edit3


def test_prepare_gl_viewer_compat_disables_newton_cuda_gl_interop_on_wsl(monkeypatch):
    _install_fake_pyglet_context(monkeypatch, lambda: None, object())
    imgui = SimpleNamespace(ImVec4=type("FakeImVec4", (), {}), color_edit3=lambda *args, **kwargs: None)
    newton_opengl = SimpleNamespace(ENABLE_CUDA_INTEROP=True)

    monkeypatch.setitem(sys.modules, "imgui_bundle", SimpleNamespace(imgui=imgui))
    monkeypatch.setitem(sys.modules, "newton._src.viewer.gl.opengl", newton_opengl)
    monkeypatch.setenv("WSL_INTEROP", "/run/WSL/123_interop")
    monkeypatch.delenv("PROTOMOTIONS_DISABLE_NEWTON_CUDA_GL_INTEROP", raising=False)

    prepare_gl_viewer_compat()

    assert newton_opengl.ENABLE_CUDA_INTEROP is False


def test_prepare_gl_viewer_compat_respects_interop_override(monkeypatch):
    _install_fake_pyglet_context(monkeypatch, lambda: None, object())
    imgui = SimpleNamespace(ImVec4=type("FakeImVec4", (), {}), color_edit3=lambda *args, **kwargs: None)
    newton_opengl = SimpleNamespace(ENABLE_CUDA_INTEROP=True)

    monkeypatch.setitem(sys.modules, "imgui_bundle", SimpleNamespace(imgui=imgui))
    monkeypatch.setitem(sys.modules, "newton._src.viewer.gl.opengl", newton_opengl)
    monkeypatch.setenv("WSL_INTEROP", "/run/WSL/123_interop")
    monkeypatch.setenv("PROTOMOTIONS_DISABLE_NEWTON_CUDA_GL_INTEROP", "0")

    prepare_gl_viewer_compat()

    assert newton_opengl.ENABLE_CUDA_INTEROP is True
