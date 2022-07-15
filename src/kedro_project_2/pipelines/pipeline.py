"""
This is a boilerplate pipeline
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_audit_report


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_audit_report,
                inputs=["staging_client", "staging_crm_call_center_logs", "staging_loan"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="generate_audit_report",
            ),
        ]
    )
