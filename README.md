# exteval-corrector
Repo for my master seminar project + paper based on the ExtEval Metric

### Requirements

* Install Docker Container 
* Build Docker: run build -t exteval-corrector .
* Run Docker: docker run -p 8888:8888 exteval-corrector
* Open Jupyter Notebook in Browser: http://localhost:8888 (use token from console output)

### All information are present in the notebooks -> have fun exploring!

### Structure

* `.ipynb_checkpoints/`: contains checkpoints of the notebooks (shows the last time the notebook was run with the output)
* `data/`: contains the data used in the project
  * `corrected/`: contains the corrected data
    * `preprocessed/`: contains the preprocessed data (for the corrected summaries to use in the ExtEval (and ExtEval-Modified) metric)
    * `scores/`: contains the ExtEval scores of the corrected data (and ExtEval-Modified scores)
    * `corrected_data.json`: contains the corrected data with the ExtEval scores
    * `corrected_data_modified.json`: contains the corrected data with the ExtEval-Modified scores
  * `merged/`: contains the merged data (data + ExtEval (ExtEval-Modified) scores)
  * `scores/`: contains the ExtEval and ExtEval-Modified scores of the incorrect data
  * `data.json`: contains the raw data
  * `data_only_incorrect.json`: contains only the incorrect entries in the raw data
* `exteval/`: contains the implementation of the ExtEval and the ExtEval-Modified metric
* `results/`: contains the results of the experiments
  * `comparison_statistical_summaries.csv`: contains the statistical comparison of the ExtEval and ExtEval-Modified metric results
  * `percentage_improvement_comparison.csv`: contains the percentage improvement of the ExtEval metric for every metric and every entry in the data
    * here the SentiBias outliers can be seen
  * `percentage_improvement_comparison_modified.csv`: contains the percentage improvement of the ExtEval-Modified metric for every metric and every entry in the data
    * here the SentiBias outliers can be seen
  * ...
* `exteval-corrector-research.ipynb`: contains the developed ExtEval-Corrector framework with all steps and explanations
* `exteval-modified-corrector-research.ipynb`: contains the developed ExtEval-Modified-Corrector framework with all steps and explanations
* `research-results.ipynb`: contains the comparison results of the both frameworks


