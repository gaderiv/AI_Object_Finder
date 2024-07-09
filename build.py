from pybuilder.core import use_plugin, init
import os

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.install_dependencies")
use_plugin("python.distutils")

name = "AI_Object_Finder"
default_task = "publish"

@init
def set_properties(project):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_main_python = os.path.join(base_dir, "src", "main", "python")
    src_unittest_python = os.path.join(base_dir, "src", "test", "python")
    src_main_resources = os.path.join(base_dir, "src", "main", "resources")
    target_dir = os.path.join(base_dir, "target")
    
    project.set_property("dir_source_main_python", src_main_python)
    project.set_property("dir_source_unittest_python", src_unittest_python)
    project.set_property("dir_source_main_resources", src_main_resources)
    project.set_property("dir_target", target_dir)
    project.set_property("unittest_module_glob", "test_*")

    project.build_depends_on("torch")
    project.build_depends_on("torchvision")
    project.build_depends_on("efficientnet_pytorch")
    project.build_depends_on("numpy")
    project.build_depends_on("opencv-python")
    project.build_depends_on("tk")
