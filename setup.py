from setuptools import setup, find_packages

setup(
    name="SCUD",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "datasets",
        "huggingface_hub",
        "lightning",
        "einops",
        "hydra-core",
        "omegaconf",
        "wandb",
        "transformers",
        "faiss-cpu",
        "pytorch-lightning",
        "evodiff",
    ],
    python_requires='>=3.8.5',
)