import mlflow
from mlflow import log_metric, log_param, log_artifact, set_tag

if __name__ == "__main__":

    tracking_uri = '/Users/takapy/python/competition/mlflow/mlruns'
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-experiment")
    mlflow.start_run(run_name='run_name001')

    # Log a parameter (key-value pair)
    log_param('param1', 42)

    # Log a metric; metrics can be updated throughout the run
    log_metric('fold1_score', 9.99)
    log_metric('fold2_score', 9.92)
    log_metric('fold3_score', 9.78)

    # Log an artifact (output file)
    with open("output.txt", "w") as f:
        f.write("Hello world sample!")

    log_artifact("output.txt")

    set_tag('tag1', 'this is tag1')
    set_tag('tag2', 'this is tag2')

    mlflow.end_run()
