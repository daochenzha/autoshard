import setuptools

setuptools.setup(
    name="autoshard",
    version='1.0.0',
    author="Daochen Zha",
    author_email="daochen.zha@rice.edu",
    description="autoshard",
    url="https://github.com/daochenzha/autoshard",
    keywords=["Reinforcement Learning"],
    packages=setuptools.find_packages(exclude=('tests',)),
    requires_python='>=3.8',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
