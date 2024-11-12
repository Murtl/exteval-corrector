# Verwende ein offizielles AllenNLP-Image als Basis
FROM allennlp/allennlp

# Arbeitsverzeichnis für den Container festlegen
WORKDIR /app

# Installiere spezifische Projektabhängigkeiten
RUN pip install torch==1.12.1 \
    transformers==4.19.2 \
    tqdm \
    nltk \
    allennlp-models

# Kopiere alle Dateien des aktuellen Verzeichnisses in das Arbeitsverzeichnis des Containers
COPY . /app

# Default command to run when the container starts (kann angepasst werden)
ENTRYPOINT ["/bin/bash"]


