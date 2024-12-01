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
    openai \
    matplotlib

# copy the current directory contents into the container at /app
COPY data /app/data
COPY exteval /app/exteval
COPY results /app/results
COPY exteval-corrector-research.ipynb /app
COPY exteval-modified-corrector-research.ipynb /app
COPY reserach-results.ipynb /app
# COPY resumable_train.sh /app # after you executed the preprocess or exteval script this makes sense to use for the next docker build & docker run (copy it out of docker before)

# copy root and stage for caches
# COPY root /root # after you executed the preprocess or exteval script this makes sense to use for the next docker build & docker run (copy it out of docker before)
# COPY stage /stage # after you executed the preprocess or exteval script this makes sense to use for the next docker build & docker run (copy it out of docker before)

# Expose port 8888 to access Jupyter Lab
EXPOSE 8888

# Default command to run when the container starts (kann angepasst werden)
#ENTRYPOINT ["/bin/bash"]

# start command for Jupyter Lab
ENTRYPOINT []
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

