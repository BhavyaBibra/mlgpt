from setuptools import find_packages, setup

setup(
    name="mlgpt",
    version="0.1.0",
    description="Ask your ML models why they're failing.",
    packages=find_packages(),
    install_requires=["httpx>=0.27"],
    python_requires=">=3.10",
)
