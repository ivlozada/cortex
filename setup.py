from setuptools import setup, find_packages

# Leemos el README para que la descripción en PyPI sea la misma que en GitHub
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cortex",  # El nombre corto y duro.
    version="0.1.0-alpha", # Empezamos con Alpha para denotar "Early Access" exclusivo
    author="Ivan Lozada", # Tu nombre o el de tu organización
    author_email="ilozada@example.com", # Pon tu email real si vas a publicar
    description="The Epistemic Inference Engine for Python. No prompts. Just logic.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/cortex", # URL de tu repo
    packages=find_packages(), # Esto encuentra automáticamente cortex/ core/ io/ api/
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.7',
    install_requires=[
        # Aquí irían dependencias. 
        # Como prometimos "CPU Pura", mantenemos esto vacío o mínimo.
        # Si usaste numpy en alguna parte interna, agrégalo aquí:
        # "numpy>=1.21.0", 
    ],
    entry_points={
        # Esto permitiría ejecutar cortex desde la terminal directamente en el futuro
        # 'console_scripts': [
        #     'cortex=cortex.cli:main',
        # ],
    },
)
