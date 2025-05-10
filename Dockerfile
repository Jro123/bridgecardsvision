FROM ubuntu:20.04

# Installer les dépendances
RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    libopencv-dev \
    python3.12 \
    python3.12-dev \
    libcurl4-openssl-dev \
    && apt-get clean

# Copier le code source dans le conteneur
COPY . /app
WORKDIR /app

# Construire le projet
RUN mkdir -p build && cd build && cmake .. && make

# Définir la commande par défaut
CMD ["./build/cv2c"]

