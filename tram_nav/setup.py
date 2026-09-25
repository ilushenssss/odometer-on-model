from glob import glob

from setuptools import find_packages, setup

package_name = "tram_nav"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="tram_nav maintainers",
    maintainer_email="ibel71531@gmail.com",
    description="GNSS-free model-based navigation of a tram from controller handle and odometry",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "navigator = tram_nav.nodes.navigator_node:main",
            "simulator = tram_nav.nodes.simulator_node:main",
            "evaluator = tram_nav.nodes.evaluator_node:main",
        ],
    },
)
