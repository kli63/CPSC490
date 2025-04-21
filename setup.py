from setuptools import setup, find_packages

setup(
    name="cpsc490",
    version="0.1.0",
    description="Neural 3D Modeling and Rendering",
    author="Yale CPSC490",
    packages=find_packages(),
    install_requires=[
        # Core dependencies provided by requirements.txt
    ],
    scripts=[
        "src/scripts/shapegen.py",
        "src/scripts/render_video.py",
        "src/scripts/render_sphere_views.py"
    ]
)