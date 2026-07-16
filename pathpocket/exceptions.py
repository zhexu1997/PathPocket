"""
Exceptions for PathPocket
"""


class PipelineNotInitializedError(KeyError):
    """Raised when pipeline status is accessed before initialization."""

    def __init__(self, namespace: str = ""):
        msg = (
            f"Pipeline namespace '{namespace}' not found.\n"
            f"Please call initialize_pipeline_status() before accessing pipeline status."
        )
        super().__init__(msg)
        self.namespace = namespace
