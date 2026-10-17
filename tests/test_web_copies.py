import filecmp
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_web_app_copies_match_the_library():
    # the web app is deployed without the metaforge package; scripts/export_web_model.py copies these
    for name in ("strength_edge.py", "phase.py"):
        src = os.path.join(ROOT, "metaforge", name)
        dst = os.path.join(ROOT, "MetaForge-Web", "metaforge_core", name)
        assert os.path.exists(dst), f"run scripts/export_web_model.py ({name} missing)"
        assert filecmp.cmp(src, dst, shallow=False), f"{name} differs; rerun scripts/export_web_model.py"
