# use the official AllenNLP image as a base
FROM allennlp/allennlp

# working directory inside the container
WORKDIR /app

# install torch, transformers, tqdm, nltk, allennlp-models, jupyterlab, openai
RUN pip install torch==1.12.1 \
    transformers==4.19.2 \
    tqdm \
    nltk \
    allennlp-models \
    jupyterlab \
    openai

# copy the current directory contents into the container at /app
COPY data /app/data
COPY exteval /app/exteval
COPY exteval-corrector.ipynb /app

# copy root and stage for caches
COPY root /root
COPY stage /stage

# Expose port 8888 to access Jupyter Lab
EXPOSE 8888

# Default command to run when the container starts (kann angepasst werden)
#ENTRYPOINT ["/bin/bash"]

# start command for Jupyter Lab
ENTRYPOINT []
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

